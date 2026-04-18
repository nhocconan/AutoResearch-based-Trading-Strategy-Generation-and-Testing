# 1d Weekly Donchian Breakout with Volume Confirmation
# Hypothesis: On 1d timeframe, breaking above the weekly Donchian high (20-day) or below the weekly Donchian low 
# with volume confirmation captures major trend moves. Weekly timeframe reduces noise and false breakouts, 
# while volume confirmation ensures institutional participation. Works in bull markets via upward breaks and 
# in bear markets via downward breaks. Low trade frequency due to weekly breakout requirement.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Weekly Donchian channels (20-period high/low)
    weekly_high = pd.Series(df_weekly['high'].values).rolling(window=20, min_periods=20).max().values
    weekly_low = pd.Series(df_weekly['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to daily timeframe
    weekly_high_aligned = align_htf_to_ltf(prices, df_weekly, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_weekly, weekly_low)
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Warmup for 20-period calculations (need 20+20 for alignment)
    
    for i in range(start_idx, n):
        if np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        donchian_high = weekly_high_aligned[i]
        donchian_low = weekly_low_aligned[i]
        vol_ok = vol_confirm[i]
        
        if position == 0:
            # Enter long: price breaks above weekly Donchian high with volume
            if close[i] > donchian_high and vol_ok:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below weekly Donchian low with volume
            elif close[i] < donchian_low and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below weekly Donchian high
            if close[i] < donchian_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above weekly Donchian low
            if close[i] > donchian_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Weekly_Donchian_Breakout_Volume"
timeframe = "1d"
leverage = 1.0