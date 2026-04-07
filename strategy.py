#!/usr/bin/env python3
"""
1d_aroon_oscillator_weekly_trend_volume_v1
Hypothesis: Aroon Oscillator (25) on 1d identifies trend strength. When AO > 0 (uptrend) and price above weekly EMA10 with volume confirmation, go long. When AO < 0 (downtrend) and price below weekly EMA10 with volume confirmation, go short. Uses weekly trend filter to avoid whipsaws, targeting 10-25 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_aroon_oscillator_weekly_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Weekly EMA10 for trend filter
    weekly_ema10 = df_weekly['close'].ewm(span=10, adjust=False).mean()
    weekly_ema10_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema10.values)
    
    # Aroon Oscillator (25) on daily
    # Aroon Up = ((Period - Days Since Highest High) / Period) * 100
    # Aroon Down = ((Period - Days Since Lowest Low) / Period) * 100
    # Aroon Oscillator = Aroon Up - Aroon Down
    period = 25
    aroon_up = np.full(n, np.nan)
    aroon_down = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        highest_high_idx = np.argmax(high[i - period + 1:i + 1])
        lowest_low_idx = np.argmin(low[i - period + 1:i + 1])
        days_since_high = period - 1 - highest_high_idx
        days_since_low = period - 1 - lowest_low_idx
        aroon_up[i] = ((period - days_since_high) / period) * 100
        aroon_down[i] = ((period - days_since_low) / period) * 100
    
    aroon_osc = aroon_up - aroon_down
    
    # Volume confirmation (20-day average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(aroon_osc[i]) or np.isnan(weekly_ema10_aligned[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: Aroon Oscillator <= 0 or price below weekly EMA10
            if aroon_osc[i] <= 0 or close[i] < weekly_ema10_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: Aroon Oscillator >= 0 or price above weekly EMA10
            if aroon_osc[i] >= 0 or close[i] > weekly_ema10_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Aroon Oscillator > 0, with volume and price above weekly EMA10
            if (aroon_osc[i] > 0 and vol_confirm and 
                close[i] > weekly_ema10_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Aroon Oscillator < 0, with volume and price below weekly EMA10
            elif (aroon_osc[i] < 0 and vol_confirm and 
                  close[i] < weekly_ema10_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals