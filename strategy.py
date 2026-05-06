#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Donchian(20) breakout with volume confirmation and ATR-based volatility filter
# Long when price breaks above 1w Donchian upper band AND volume > 1.5 * 20-period avg volume AND ATR(14) < ATR(50) (low vol regime)
# Short when price breaks below 1w Donchian lower band with same conditions
# Exit when price crosses 1w Donchian middle band (mean reversion to equilibrium)
# Uses discrete sizing 0.25 to limit drawdown and reduce fee churn
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe
# 1w Donchian provides weekly structure, volume confirms participation, ATR filter avoids choppy markets

name = "1d_1wDonchian20_VolumeSpike_ATRRegime_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for Donchian channels and ATR regime
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Donchian(20) channels
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window-1, len(arr)):
            result[i] = np.max(arr[i-window+1:i+1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window-1, len(arr)):
            result[i] = np.min(arr[i-window+1:i+1])
        return result
    
    donchian_upper_1w = rolling_max(high_1w, 20)
    donchian_lower_1w = rolling_min(low_1w, 20)
    donchian_middle_1w = (donchian_upper_1w + donchian_lower_1w) / 2.0
    
    # Calculate 1w ATR for volatility regime filter
    def calculate_atr(high, low, close, period=14):
        tr1 = np.abs(high - low)
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        atr = np.full_like(tr, np.nan, dtype=float)
        atr[period-1] = np.mean(tr[:period])
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr_14_1w = calculate_atr(high_1w, low_1w, close_1w, period=14)
    atr_50_1w = calculate_atr(high_1w, low_1w, close_1w, period=50)
    atr_regime_1w = atr_14_1w < atr_50_1w  # Low volatility regime
    
    # Get 1d data ONCE before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume confirmation: volume > 1.5 * 20-period average volume
    avg_volume_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (1.5 * avg_volume_20_1d)
    
    # Align 1w indicators to 1d timeframe (wait for completed 1w bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper_1w)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower_1w)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_1w, donchian_middle_1w)
    atr_regime_aligned = align_htf_to_ltf(prices, df_1w, atr_regime_1w)
    
    # Align 1d indicators to 1d timeframe (no delay needed for same timeframe)
    volume_spike_aligned = volume_spike_1d
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(atr_regime_aligned[i]) or 
            np.isnan(volume_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1w Donchian upper with volume spike and low vol regime
            if (close[i] > donchian_upper_aligned[i] and close[i-1] <= donchian_upper_aligned[i-1] and 
                volume_spike_aligned[i] and atr_regime_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1w Donchian lower with volume spike and low vol regime
            elif (close[i] < donchian_lower_aligned[i] and close[i-1] >= donchian_lower_aligned[i-1] and 
                  volume_spike_aligned[i] and atr_regime_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 1w Donchian middle (mean reversion)
            if close[i] < donchian_middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 1w Donchian middle (mean reversion)
            if close[i] > donchian_middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals