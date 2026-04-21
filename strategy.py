#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1d EMA34 trend filter and volume confirmation.
# Long when Williams %R crosses above -20 from below in uptrend (price > 1d EMA34).
# Short when Williams %R crosses below -80 from above in downtrend (price < 1d EMA34).
# Exit on opposite Williams %R cross (-80 for long, -20 for short).
# Uses volume > 1.2x 20-period average for confirmation.
# Target: 20-40 trades/year by requiring trend alignment + momentum extreme + volume confirmation.
# Williams %R identifies overbought/oversold conditions; EMA34 filters trend direction.

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    # Calculate Williams %R (14-period)
    highest_high = prices['high'].rolling(window=14, min_periods=14).max()
    lowest_low = prices['low'].rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - prices['close']) / (highest_high - lowest_low)
    williams_r = williams_r.replace([np.inf, -np.inf], np.nan).fillna(-50).values
    
    # Calculate 1d EMA34 trend
    df_1d = get_htf_data(prices, '1d')
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Pre-compute volume moving average (20-period)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):
        # Skip if data not ready
        if np.isnan(williams_r[i]) or np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.2x 20-period average
        volume_confirm = volume > 1.2 * vol_ma[i]
        
        # Trend filter
        uptrend = price > ema_34_aligned[i]
        downtrend = price < ema_34_aligned[i]
        
        # Williams %R levels
        wr = williams_r[i]
        wr_prev = williams_r[i-1] if i > 0 else -50
        
        if position == 0:
            if volume_confirm:
                # Long: Williams %R crosses above -20 from below in uptrend
                if wr_prev <= -20 and wr > -20 and uptrend:
                    signals[i] = 0.25
                    position = 1
                # Short: Williams %R crosses below -80 from above in downtrend
                elif wr_prev >= -80 and wr < -80 and downtrend:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit long when Williams %R crosses below -80
                if wr_prev >= -80 and wr < -80:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit short when Williams %R crosses above -20
                if wr_prev <= -20 and wr > -20:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsR_1dEMA34_Trend_Volume"
timeframe = "12h"
leverage = 1.0