#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian(20) breakout with 1d volume spike and ATR regime filter
# Long when price breaks above 4h Donchian upper band AND 1d volume > 2.0 * avg_volume(20) AND ATR(14) < ATR(50) (low volatility regime)
# Short when price breaks below 4h Donchian lower band AND 1d volume > 2.0 * avg_volume(20) AND ATR(14) < ATR(50)
# Exit when price crosses 4h Donchian middle band (mean reversion to equilibrium)
# Uses discrete sizing 0.20 to balance return and drawdown
# Target: 60-150 total trades over 4 years = 15-37/year for 1h timeframe
# 4h Donchian provides 4h structure with proven breakout edge
# Volume spike confirms participation (reduces false breakouts)
# Low volatility regime (ATR14 < ATR50) filters choppy markets and focuses on explosive moves
# Session filter (08-20 UTC) to reduce noise trades

name = "1h_4hDonchian20_1dVolumeSpike_ATRRegime_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data ONCE before loop for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:  # Need sufficient data for Donchian and ATR
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Donchian(20) channels based on previous 4h bar
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
    
    donchian_upper_4h = rolling_max(high_4h, 20)
    donchian_lower_4h = rolling_min(low_4h, 20)
    donchian_middle_4h = (donchian_upper_4h + donchian_lower_4h) / 2.0
    
    # Calculate 4h ATR for volatility regime filter
    def calculate_atr(high, low, close, period=14):
        # True Range
        tr1 = np.abs(high - low)
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First period TR is just high-low
        
        # Wilder's smoothing (equivalent to EMA with alpha=1/period)
        atr = np.full_like(tr, np.nan, dtype=float)
        atr[period-1] = np.mean(tr[:period])
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr_14_4h = calculate_atr(high_4h, low_4h, close_4h, period=14)
    atr_50_4h = calculate_atr(high_4h, low_4h, close_4h, period=50)
    atr_regime_4h = atr_14_4h < atr_50_4h  # Low volatility regime
    
    # Get 1d data ONCE before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for volume average
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume confirmation: volume > 2.0 * 20-period average volume
    avg_volume_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (2.0 * avg_volume_20_1d)
    
    # Align 4h indicators to 1h timeframe (wait for completed 4h bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper_4h)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower_4h)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_4h, donchian_middle_4h)
    atr_regime_aligned = align_htf_to_ltf(prices, df_4h, atr_regime_4h)
    
    # Align 1d indicators to 1h timeframe (wait for completed 1d bar)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(atr_regime_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 4h Donchian upper with volume spike and low vol regime
            if (close[i] > donchian_upper_aligned[i] and close[i-1] <= donchian_upper_aligned[i-1] and 
                volume_spike_aligned[i] and atr_regime_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h Donchian lower with volume spike and low vol regime
            elif (close[i] < donchian_lower_aligned[i] and close[i-1] >= donchian_lower_aligned[i-1] and 
                  volume_spike_aligned[i] and atr_regime_aligned[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price crosses below 4h Donchian middle (mean reversion)
            if close[i] < donchian_middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price crosses above 4h Donchian middle (mean reversion)
            if close[i] > donchian_middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals