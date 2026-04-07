#!/usr/bin/env python3
"""
1d_weekly_keltner_breakout_volume_v1
Hypothesis: On 1d timeframe, enter long when price breaks above upper Keltner Channel (EMA20 + 2*ATR) with above-average volume and weekly trend bullish (price above weekly EMA20), enter short when price breaks below lower Keltner Channel (EMA20 - 2*ATR) with above-average volume and weekly trend bearish (price below weekly EMA20). Exit when price crosses back inside the Keltner Channel (middle line). Designed for 7-25 trades/year to minimize fee decay while capturing breakouts in both bull and bear markets via volatility expansion.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_keltner_breakout_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate EMA20 for Keltner Channel middle
    ema_20 = pd.Series(close).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Calculate ATR(14)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Keltner Channels
    upper_keltner = ema_20 + 2 * atr
    lower_keltner = ema_20 - 2 * atr
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Weekly EMA20 for trend filter
    weekly_close = df_weekly['close'].values
    weekly_ema_20 = pd.Series(weekly_close).ewm(span=20, min_periods=20, adjust=False).mean().values
    weekly_ema_20_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(upper_keltner[i]) or np.isnan(lower_keltner[i]) or 
            np.isnan(ema_20[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(close[i]) or np.isnan(weekly_ema_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: above average volume
        vol_ok = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below middle line (EMA20)
            if close[i] < ema_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above middle line (EMA20)
            if close[i] > ema_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Long: price breaks above upper Keltner with weekly bullish trend
                if close[i] > upper_keltner[i] and close[i-1] <= upper_keltner[i-1] and close[i] > weekly_ema_20_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below lower Keltner with weekly bearish trend
                elif close[i] < lower_keltner[i] and close[i-1] >= lower_keltner[i-1] and close[i] < weekly_ema_20_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals