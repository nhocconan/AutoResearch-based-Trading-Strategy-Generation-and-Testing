#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian channel breakout with 1d ATR volatility filter and volume confirmation
# - Long when price breaks above 20-period Donchian high AND 1d ATR(14) > 20-period SMA AND volume > 1.5x 20-period average
# - Short when price breaks below 20-period Donchian low AND 1d ATR(14) > 20-period SMA AND volume > 1.5x 20-period average
# - Exit when price returns to the midpoint of the Donchian channel (mean reversion)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Donchian breakouts capture trending moves, ATR filter ensures sufficient volatility, volume confirmation reduces false signals
# - Works in both bull and bear markets by capturing momentum bursts during ranging periods
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)

name = "12h_1d_donchian_atr_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute 12h OHLC and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 12h Donchian channel (20-period)
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.max(arr[i - window + 1:i + 1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.min(arr[i - window + 1:i + 1])
        return result
    
    donch_high = rolling_max(high, 20)
    donch_low = rolling_min(low, 20)
    donch_mid = (donch_high + donch_low) / 2
    
    # Pre-compute 12h volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 12h ATR (14-period) for volatility filter
    def true_range(h, l, c_prev):
        tr1 = h - l
        tr2 = np.abs(h - c_prev)
        tr3 = np.abs(l - c_prev)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    tr = np.zeros_like(high)
    tr[0] = high[0] - low[0]  # First bar
    for i in range(1, len(high)):
        tr[i] = true_range(high[i], low[i], close[i-1])
    
    atr = np.zeros_like(tr)
    atr[13] = np.mean(tr[1:15])  # First ATR value
    for i in range(14, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14  # Wilder's smoothing
    
    atr_ma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    volatility_filter = atr > atr_ma  # ATR above its 20-period average
    
    # Pre-compute 1d ATR (14-period) for HTF volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr_1d = np.zeros_like(high_1d)
    tr_1d[0] = high_1d[0] - low_1d[0]  # First bar
    for i in range(1, len(high_1d)):
        tr_1d[i] = true_range(high_1d[i], low_1d[i], close_1d[i-1])
    
    atr_1d = np.zeros_like(tr_1d)
    atr_1d[13] = np.mean(tr_1d[1:15])  # First ATR value
    for i in range(14, len(tr_1d)):
        atr_1d[i] = (atr_1d[i-1] * 13 + tr_1d[i]) / 14  # Wilder's smoothing
    
    # Align HTF indicators to 12h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_1d_ma = pd.Series(atr_1d_aligned).rolling(window=20, min_periods=20).mean().values
    volatility_filter_1d = atr_1d_aligned > atr_1d_ma  # 1d ATR above its 20-period average
    
    # Volume spike and volatility filter aligned to 12h
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    volatility_filter_aligned = align_htf_to_ltf(prices, df_1d, volatility_filter)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(donch_mid[i]) or 
            np.isnan(volume_spike_aligned[i]) or 
            np.isnan(volatility_filter_1d[i]) or 
            np.isnan(volatility_filter_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above Donchian high AND volume spike AND volatility filter (both TFs)
            if (close[i] > donch_high[i] and 
                volume_spike_aligned[i] and 
                volatility_filter_1d[i] and 
                volatility_filter_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below Donchian low AND volume spike AND volatility filter (both TFs)
            elif (close[i] < donch_low[i] and 
                  volume_spike_aligned[i] and 
                  volatility_filter_1d[i] and 
                  volatility_filter_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price returns to the midpoint of the Donchian channel (mean reversion)
            exit_long = (position == 1 and close[i] < donch_mid[i])
            exit_short = (position == -1 and close[i] > donch_mid[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals