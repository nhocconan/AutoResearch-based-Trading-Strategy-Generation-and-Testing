#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1d EMA(34) trend filter and volume spike.
# Long when Williams %R crosses above -20 from below (oversold reversal) in uptrend (price > EMA34).
# Short when Williams %R crosses below -80 from above (overbought reversal) in downtrend (price < EMA34).
# Volume > 2.0x 20-period average confirms reversal strength.
# Williams %R identifies exhaustion points; EMA34 filters trend direction to avoid counter-trend trades.
# Volume spike ensures momentum behind the reversal. Target: 20-40 trades/year.
# Works in bull/bear: EMA34 trend filter ensures trades align with higher-timeframe momentum.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA(34) for trend direction
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate 14-period Williams %R on 12h data
    high_14 = prices['high'].rolling(window=14, min_periods=14).max()
    low_14 = prices['low'].rolling(window=14, min_periods=14).min()
    close = prices['close']
    williams_r = -100 * (high_14 - close) / (high_14 - low_14)
    williams_r = williams_r.replace([np.inf, -np.inf], np.nan).fillna(0).values
    
    # Pre-compute volume moving average (20-period)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        # Skip if data not ready
        if np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(williams_r[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Williams %R values
        wr = williams_r[i]
        wr_prev = williams_r[i-1] if i > 0 else wr
        
        # Volume confirmation: current volume > 2.0x 20-period average
        volume_confirm = volume > 2.0 * vol_ma[i]
        
        # Trend filter: price relative to EMA34
        uptrend = price > ema_34_aligned[i]
        downtrend = price < ema_34_aligned[i]
        
        if position == 0:
            if volume_confirm:
                # Long: Williams %R crosses above -20 from below in uptrend
                if wr > -20 and wr_prev <= -20 and uptrend:
                    signals[i] = 0.25
                    position = 1
                # Short: Williams %R crosses below -80 from above in downtrend
                elif wr < -80 and wr_prev >= -80 and downtrend:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if Williams %R rises above -20 (overbought) or trend changes
                if wr > -20 or not uptrend:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if Williams %R falls below -80 (oversold) or trend changes
                if wr < -80 or not downtrend:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsR14_1dEMA34_Trend_Volume"
timeframe = "12h"
leverage = 1.0