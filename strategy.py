#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d HMA21 trend filter and volume confirmation.
# Enter long when price breaks above Donchian upper band with 1d HMA21 uptrend and volume > 1.5x 20-bar average.
# Enter short when price breaks below Donchian lower band with 1d HMA21 downtrend and volume > 1.5x 20-bar average.
# Exit when price retraces to the Donchian midline (average of upper and lower bands).
# Uses discrete position sizing (0.25) to limit drawdown and reduce fee churn.
# Target: 75-200 total trades over 4 years (19-50/year).
# Donchian provides clear structure; 1d HMA21 ensures higher timeframe alignment; volume spike filters weak breakouts.
# Works in both bull (strong breakouts) and bear (strong breakdowns) due to symmetric long/short logic.

name = "4h_Donchian20_Breakout_1dHMA21_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for HMA21 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 21:
        return np.zeros(n)
    
    # Calculate 1d HMA21
    close_1d = df_1d['close'].values
    half_length = 21 // 2
    sqrt_length = int(np.sqrt(21))
    
    # WMA function
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, mode='valid') / weights.sum()
    
    # HMA calculation
    wma_half = np.array([np.nan] * len(close_1d))
    wma_full = np.array([np.nan] * len(close_1d))
    
    for i in range(half_length - 1, len(close_1d)):
        wma_half[i] = wma(close_1d[i - half_length + 1:i + 1], half_length)
    
    for i in range(21 - 1, len(close_1d)):
        wma_full[i] = wma(close_1d[i - 21 + 1:i + 1], 21)
    
    hma_raw = 2 * wma_half - wma_full
    hma_21 = np.array([np.nan] * len(close_1d))
    
    for i in range(sqrt_length - 1, len(hma_raw)):
        if not np.isnan(hma_half[i - sqrt_length + 1:i + 1]).any():
            hma_21[i] = wma(hma_half[i - sqrt_length + 1:i + 1], sqrt_length)
    
    # Align HMA21 to 4h
    hma_21_aligned = align_htf_to_ltf(prices, df_1d, hma_21)
    
    # Get 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate Donchian(20) on 4h
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    donch_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # Align Donchian to 4h (no additional delay needed as we use completed 4h bars)
    donch_high_aligned = align_htf_to_ltf(prices, df_4h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_4h, donch_low)
    donch_mid_aligned = align_htf_to_ltf(prices, df_4h, donch_mid)
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure sufficient history for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(hma_21_aligned[i]) or np.isnan(volume_ma_20[i]) or 
            np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(donch_mid_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # 1d HMA21 trend: slope over 3 periods
        if i >= 3:
            hma_slope = (hma_21_aligned[i] - hma_21_aligned[i-3]) / 3
            hma_trend_up = hma_slope > 0
            hma_trend_down = hma_slope < 0
        else:
            hma_trend_up = False
            hma_trend_down = False
        
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: price > Donchian high, HMA21 up, volume confirm
            if price > donch_high_aligned[i] and hma_trend_up and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: price < Donchian low, HMA21 down, volume confirm
            elif price < donch_low_aligned[i] and hma_trend_down and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - hold or exit at Donchian mid
            if price <= donch_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - hold or exit at Donchian mid
            if price >= donch_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals