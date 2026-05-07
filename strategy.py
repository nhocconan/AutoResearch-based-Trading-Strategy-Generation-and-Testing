#!/usr/bin/env python3
"""
4h_Keltner_Channel_Breakout_1dTrend_Volume_v1
Hypothesis: On 4h timeframe, use Keltner Channel breakout combined with 1d EMA trend filter and volume spikes.
Long when price breaks above upper Keltner Channel (EMA10 + 2*ATR) with price above daily EMA34 and volume spike.
Short when price breaks below lower Keltner Channel (EMA10 - 2*ATR) with price below daily EMA34 and volume spike.
Keltner Channels adapt to volatility, reducing false breakouts in low volatility periods.
Volume spikes confirm institutional interest. Works in both bull and bear markets by aligning with daily trend.
"""
name = "4h_Keltner_Channel_Breakout_1dTrend_Volume_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Keltner Channel components
    # EMA(10) for midline
    ema_10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # ATR(10) for channel width
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Upper and lower channels
    upper_channel = ema_10 + (2.0 * atr)
    lower_channel = ema_10 - (2.0 * atr)
    
    # Daily EMA34 for trend filter
    ema_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume filter: current volume > 1.8 * 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = max(34, 20, 10)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        bars_since_entry += 1
        
        # Skip if any data is not ready
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        if position == 0:
            # Minimum 6 bars between trades to reduce frequency (4h timeframe)
            if bars_since_entry < 6:
                continue
                
            # Long: break above upper channel + price above EMA34 + volume filter
            if (close[i] > upper_channel[i] and 
                close[i] > ema_34_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: break below lower channel + price below EMA34 + volume filter
            elif (close[i] < lower_channel[i] and 
                  close[i] < ema_34_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        elif position != 0:
            # Exit: price crosses back below/above midline (EMA10)
            if position == 1:
                if close[i] < ema_10[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > ema_10[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals