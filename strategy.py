#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Donchian channel breakout with volume confirmation and chop regime filter
# Long when price breaks above 12h Donchian(20) high AND chop > 61.8 (range) AND volume > 1.3 * avg_volume(20)
# Short when price breaks below 12h Donchian(20) low AND chop > 61.8 (range) AND volume > 1.3 * avg_volume(20)
# Exit when price crosses 12h EMA50 (trend filter reversal)
# Uses discrete sizing 0.25 to balance return and drawdown
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# 12h Donchian provides clear structure with proven breakout edge in ranging markets
# Chop filter ensures we only trade in ranging regimes (avoid trending whipsaws)
# Volume confirmation filters weak breakouts

name = "4h_12hDonchian20_ChopRange_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for Donchian channels and EMA
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need sufficient data for Donchian(20) and EMA50
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Donchian(20) channels based on previous 12h bar
    high_series_12h = pd.Series(high_12h)
    low_series_12h = pd.Series(low_12h)
    donchian_high_12h = high_series_12h.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low_12h = low_series_12h.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate 12h EMA50 for trend filter
    close_series_12h = pd.Series(close_12h)
    ema_50_12h = close_series_12h.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate chopiness index on 12h for regime filter (range when chop > 61.8)
    atr_period = 14
    tr1 = pd.Series(high_12h - low_12h)
    tr2 = pd.Series(np.abs(high_12h - np.roll(close_12h, 1)))
    tr3 = pd.Series(np.abs(low_12h - np.roll(close_12h, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_12h = tr.rolling(window=atr_period, min_periods=atr_period).mean().values
    
    max_high_12h = pd.Series(high_12h).rolling(window=atr_period, min_periods=atr_period).max().values
    min_low_12h = pd.Series(low_12h).rolling(window=atr_period, min_periods=atr_period).min().values
    
    # Chop = 100 * log10(sum(atr) / (max_high - min_low)) / log10(atr_period)
    sum_atr = pd.Series(atr_12h).rolling(window=atr_period, min_periods=atr_period).sum().values
    chop = 100 * (np.log10(sum_atr) - np.log10(max_high_12h - min_low_12h)) / np.log10(atr_period)
    chop_range = chop > 61.8  # Range regime
    
    # Get volume confirmation: volume > 1.3 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * avg_volume_20)
    
    # Align 12h indicators to 4h timeframe (wait for completed 12h bar)
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high_12h)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low_12h)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    chop_range_aligned = align_htf_to_ltf(prices, df_12h, chop_range.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(chop_range_aligned[i]) or 
            np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 12h Donchian high AND chop range AND volume confirmation
            if (close[i] > donchian_high_aligned[i] and close[i-1] <= donchian_high_aligned[i-1] and 
                chop_range_aligned[i] > 0.5 and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12h Donchian low AND chop range AND volume confirmation
            elif (close[i] < donchian_low_aligned[i] and close[i-1] >= donchian_low_aligned[i-1] and 
                  chop_range_aligned[i] > 0.5 and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 12h EMA50 (trend filter reversal)
            if close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 12h EMA50 (trend filter reversal)
            if close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals