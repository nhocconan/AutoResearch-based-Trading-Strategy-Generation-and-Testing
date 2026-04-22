#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 250:
        return np.zeros(n)
    
    # Hypothesis: Weekly Donchian channel breakout with daily volume confirmation and 1d EMA50 trend filter
    # Weekly price channels capture major trend structure, daily volume surge confirms institutional interest
    # EMA50 filter ensures alignment with intermediate-term trend, reducing whipsaws in ranging markets
    # Works in bull markets (breakouts to new highs) and bear markets (breakdowns to new lows)
    
    # Load weekly data once
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly Donchian channels (20-period)
    high_max = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Weekly Donchian upper and lower bands
    donch_upper = high_max
    donch_lower = low_min
    
    # Align weekly Donchian to daily timeframe
    donch_upper_aligned = align_htf_to_ltf(prices, df_1w, donch_upper)
    donch_lower_aligned = align_htf_to_ltf(prices, df_1w, donch_lower)
    
    # Daily EMA50 trend filter
    close = prices['close'].values
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Daily volume confirmation (20-period volume surge)
    vol_ma20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_surge = prices['volume'].values > 2.0 * vol_ma20
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(250, n):
        # Skip if data not ready
        if (np.isnan(donch_upper_aligned[i]) or np.isnan(donch_lower_aligned[i]) or 
            np.isnan(ema_50[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above weekly Donchian upper with volume surge AND daily EMA50 uptrend
            if close[i] > donch_upper_aligned[i] and vol_surge[i] and close[i] > ema_50[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly Donchian lower with volume surge AND daily EMA50 downtrend
            elif close[i] < donch_lower_aligned[i] and vol_surge[i] and close[i] < ema_50[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to weekly Donchian middle or opposite band touch
            donch_middle = (donch_upper_aligned[i] + donch_lower_aligned[i]) / 2
            if position == 1:
                if close[i] < donch_middle:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > donch_middle:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_WeeklyDonchian_Breakout_DailyVolume_EMA50Trend_v1"
timeframe = "1d"
leverage = 1.0