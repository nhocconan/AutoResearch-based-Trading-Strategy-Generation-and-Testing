#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1d EMA34 trend filter and volume confirmation.
# Williams %R > -20 indicates overbought (short signal), < -80 indicates oversold (long signal).
# Only take signals in direction of 1d EMA34 trend to avoid counter-trend trades.
# Volume > 1.5x 20-period average confirms momentum.
# Target: 15-30 trades/year by requiring trend alignment + extreme %R + volume confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Calculate Williams %R (14-period)
    highest_high = prices['high'].rolling(window=14, min_periods=14).max()
    lowest_low = prices['low'].rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - prices['close']) / (highest_high - lowest_low)
    williams_r = williams_r.replace([np.inf, -np.inf], np.nan).fillna(-50).values
    
    # Calculate 1d EMA34 trend (HTF)
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
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume > 1.5 * vol_ma[i]
        
        # Trend direction from 1d EMA34
        uptrend = price > ema_34_aligned[i]
        downtrend = price < ema_34_aligned[i]
        
        if position == 0:
            if volume_confirm:
                # Oversold in uptrend: long
                if williams_r[i] < -80 and uptrend:
                    signals[i] = 0.25
                    position = 1
                # Overbought in downtrend: short
                elif williams_r[i] > -20 and downtrend:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when Williams %R reaches overbought or trend breaks
                if williams_r[i] > -20 or price < ema_34_aligned[i]:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when Williams %R reaches oversold or trend breaks
                if williams_r[i] < -80 or price > ema_34_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsR_1dEMA34Trend_Volume"
timeframe = "12h"
leverage = 1.0