#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h 12-hour EMA Trend Filter with 1-day Donchian Breakout and Volume Confirmation
# Uses 12h EMA to filter trend direction, then enters on 1d Donchian breakouts with volume spike
# This combines multi-timeframe trend alignment (12h) with higher-timeframe structure (1d)
# Designed to work in both bull and bear markets by only trading breakouts in the direction of the 12h trend
# Target: 75-200 total trades over 4 years (19-50/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h EMA for trend direction (updated every 12h)
    df_12h = get_htf_data(prices, '12h')
    ema_12h = pd.Series(df_12h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Get 1d Donchian levels (breakout levels)
    df_1d = get_htf_data(prices, '1d')
    donchian_high_1d = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low_1d = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().shift(1).values
    donchian_high_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_1d)
    donchian_low_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_1d)
    
    # Volume confirmation: volume > 1.5x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 40  # for EMA and Donchian calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(donchian_high_1d_aligned[i]) or 
            np.isnan(donchian_low_1d_aligned[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Trend filter: only trade in direction of 12h EMA
        if price > ema_12h_aligned[i]:
            # Uptrend: only look for long breakouts
            if position == 0:
                # Long: price breaks above 1d Donchian high with volume filter
                if price > donchian_high_1d_aligned[i] and vol > 1.5 * avg_vol[i]:
                    position = 1
                    signals[i] = position_size
                else:
                    signals[i] = 0.0
            elif position == 1:
                # Stay long
                signals[i] = position_size
            else:  # position == -1
                # Exit short and go flat
                position = 0
                signals[i] = 0.0
        else:
            # Downtrend: only look for short breakouts
            if position == 0:
                # Short: price breaks below 1d Donchian low with volume filter
                if price < donchian_low_1d_aligned[i] and vol > 1.5 * avg_vol[i]:
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
            elif position == -1:
                # Stay short
                signals[i] = -position_size
            else:  # position == 1
                # Exit long and go flat
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_12hEMA_1dDonchian_Volume_Filter"
timeframe = "4h"
leverage = 1.0