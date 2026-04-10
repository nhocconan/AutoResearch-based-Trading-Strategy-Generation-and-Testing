#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d ATR filter and volume confirmation
# - Long when price breaks above Donchian(20) upper band AND 1d ATR(14) > 20-period mean AND volume > 1.5x 20-period mean
# - Short when price breaks below Donchian(20) lower band AND 1d ATR(14) > 20-period mean AND volume > 1.5x 20-period mean
# - Exit when price returns to Donchian(20) midpoint (mean reversion)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Donchian breakouts work well in both trending and ranging markets when confirmed with volatility and volume
# - ATR filter ensures we only trade during sufficient volatility periods
# - Volume confirmation reduces false breakouts
# - Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)

name = "4h_1d_donchian_atr_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute 4h OHLC and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 4h Donchian channels (20-period)
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
    
    upper_band = rolling_max(high, 20)
    lower_band = rolling_min(low, 20)
    middle_band = (upper_band + lower_band) / 2
    
    # Pre-compute 4h volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 1d ATR (14-period) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range for 1d
    tr_1d = np.zeros_like(high_1d)
    tr_1d[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(high_1d)):
        tr1 = high_1d[i] - low_1d[i]
        tr2 = np.abs(high_1d[i] - close_1d[i-1])
        tr3 = np.abs(low_1d[i] - close_1d[i-1])
        tr_1d[i] = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate ATR (14-period) using Wilder's smoothing
    atr_1d = np.zeros_like(tr_1d)
    atr_1d[13] = np.mean(tr_1d[1:15])  # First ATR value
    for i in range(14, len(tr_1d)):
        atr_1d[i] = (atr_1d[i-1] * 13 + tr_1d[i]) / 14
    
    # Pre-compute 1d ATR 20-period mean for volatility filter
    atr_ma_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_filter_1d = atr_1d > atr_ma_1d  # Trade when ATR above its mean
    
    # Align HTF indicators to 4h timeframe
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    middle_band_aligned = align_htf_to_ltf(prices, df_1d, middle_band)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    atr_filter_aligned = align_htf_to_ltf(prices, df_1d, atr_filter_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or 
            np.isnan(middle_band_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or np.isnan(atr_filter_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above upper band AND volume spike AND ATR filter
            if (close[i] > upper_band_aligned[i] and 
                volume_spike_aligned[i] and 
                atr_filter_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below lower band AND volume spike AND ATR filter
            elif (close[i] < lower_band_aligned[i] and 
                  volume_spike_aligned[i] and 
                  atr_filter_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price returns to middle band (mean reversion)
            exit_long = (position == 1 and close[i] < middle_band_aligned[i])
            exit_short = (position == -1 and close[i] > middle_band_aligned[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals