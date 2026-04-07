#!/usr/bin/env python3
"""
1h Donchian Breakout with Volume Confirmation and 4h/1d Trend Filter
Long when price breaks above Donchian upper band (20-period) with expanding volume AND 4h/1d trend up
Short when price breaks below Donchian lower band with expanding volume AND 4h/1d trend down
Exit when price crosses back to middle of Donchian channel
Uses 4h EMA trend and 1d EMA trend for multi-timeframe confirmation.
Targets 15-37 trades per year (60-150 over 4 years) by using strict entry conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_donchian_breakout_volume_4h1d_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Donchian Channel (20-period high/low) ===
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # === Volume confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)  # Avoid division by zero
    
    # === 4h trend filter (EMA 21) ===
    df_4h = get_htf_data(prices, '4h')
    ema_4h = pd.Series(df_4h['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # === 1d trend filter (EMA 21) ===
    df_1d = get_htf_data(prices, '1d')
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === Session filter: 08-20 UTC (only trade during active hours) ===
    # Convert open_time to datetime if needed and extract hour
    if not isinstance(prices.index, pd.DatetimeIndex):
        # If index is not datetime, use the 'open_time' column
        hours = pd.to_datetime(prices['open_time']).hour.values
    else:
        hours = prices.index.hour.values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(donch_mid[i]) or np.isnan(vol_ratio[i]) or 
            np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Check session: only trade between 08:00 and 20:00 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses back below middle line
            if close[i] < donch_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price crosses back above middle line
            if close[i] > donch_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Need expanding volume (above average)
            if vol_ratio[i] < 1.5:
                signals[i] = 0.0
                continue
            
            # Entry: Donchian breakout with volume confirmation AND 4h/1d trend filter
            # Both 4h and 1d EMAs must be trending in the same direction
            if i > 0:
                ema_4h_rising = ema_4h_aligned[i] > ema_4h_aligned[i-1]
                ema_1d_rising = ema_1d_aligned[i] > ema_1d_aligned[i-1]
                ema_4h_falling = ema_4h_aligned[i] < ema_4h_aligned[i-1]
                ema_1d_falling = ema_1d_aligned[i] < ema_1d_aligned[i-1]
                
                # Long: price breaks above upper channel with rising 4h AND 1d EMA
                if close[i] > donch_high[i] and vol_ratio[i] >= 1.5 and ema_4h_rising and ema_1d_rising:
                    position = 1
                    signals[i] = 0.20
                # Short: price breaks below lower channel with falling 4h AND 1d EMA
                elif close[i] < donch_low[i] and vol_ratio[i] >= 1.5 and ema_4h_falling and ema_1d_falling:
                    position = -1
                    signals[i] = -0.20
    
    return signals