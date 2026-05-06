#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Donchian channel breakout with volume confirmation and ATR filter
# Long when price breaks above 1d Donchian(20) upper band AND volume > 1.5 * avg_volume(20) AND ATR(14) < ATR(50)
# Short when price breaks below 1d Donchian(20) lower band AND volume > 1.5 * avg_volume(20) AND ATR(14) < ATR(50)
# Exit when price returns to 1d Donchian midpoint or opposite band touched
# Uses discrete sizing 0.25 to balance return and drawdown control
# ATR filter ensures we only trade in low volatility regimes (reduces whipsaw)
# Works in both bull (continuation breakouts) and bear (continuation breakdowns) markets
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe

name = "4h_1dDonchian20_Volume_ATR_Filter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Donchian and ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need sufficient data for ATR(50)
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian channel (20-period)
    # Upper band = max(high, lookback=20)
    # Lower band = min(low, lookback=20)
    # Middle band = (upper + lower) / 2
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # Calculate ATR(14) and ATR(50) for volatility filter
    # True Range = max(high-low, abs(high-close_prev), abs(low-close_prev))
    close_shift = np.roll(close_1d, 1)
    close_shift[0] = np.nan  # First value has no previous close
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - close_shift)
    tr3 = np.abs(low_1d - close_shift)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Align 1d indicators to 4h timeframe (wait for completed 1d bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_1d, donchian_middle)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_50_aligned = align_htf_to_ltf(prices, df_1d, atr_50)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 4h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # ATR filter: trade only when short-term ATR < long-term ATR (low volatility regime)
    # Avoid whipsaw in high volatility markets
    atr_filter = atr_14_aligned < atr_50_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(atr_14_aligned[i]) or 
            np.isnan(atr_50_aligned[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1d Donchian upper band with volume confirmation and ATR filter
            if (close[i] > donchian_upper_aligned[i] and close[i-1] <= donchian_upper_aligned[i-1] and 
                volume_confirm[i] and atr_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Donchian lower band with volume confirmation and ATR filter
            elif (close[i] < donchian_lower_aligned[i] and close[i-1] >= donchian_lower_aligned[i-1] and 
                  volume_confirm[i] and atr_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to 1d Donchian middle or touches lower band (reversal)
            if close[i] <= donchian_middle_aligned[i] or close[i] <= donchian_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to 1d Donchian middle or touches upper band (reversal)
            if close[i] >= donchian_middle_aligned[i] or close[i] >= donchian_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals