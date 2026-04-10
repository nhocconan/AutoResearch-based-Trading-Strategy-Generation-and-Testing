#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian channel breakout with 1w ADX regime filter and volume confirmation
# - Long when price breaks above 20-period Donchian high AND 1d volume > 1.5x 20-period average AND 1w ADX > 25 (strong trend)
# - Short when price breaks below 20-period Donchian low AND 1d volume > 1.5x 20-period average AND 1w ADX > 25
# - Exit when price returns to 10-period Donchian midpoint (mean of 10-period high/low) or reverses to opposite Donchian band
# - Uses discrete position sizing 0.25 to limit fee churn
# - Donchian channels provide clear structural breakout levels
# - Volume confirmation reduces false breakouts in low-liquidity periods
# - Weekly ADX filter ensures we trade only when higher timeframe is trending strongly (avoids chop and sideways markets)
# - Target: 15-25 trades/year on 1d timeframe (60-100 total over 4 years)

name = "1d_1w_donchian_volume_adx_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d OHLCV
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 1d Donchian channels (20-period for entry, 10-period for exit)
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
    
    donchian_high_20 = rolling_max(high, 20)
    donchian_low_20 = rolling_min(low, 20)
    donchian_high_10 = rolling_max(high, 10)
    donchian_low_10 = rolling_min(low, 10)
    donchian_mid_10 = (donchian_high_10 + donchian_low_10) / 2.0
    
    # Pre-compute 1d volume average (20-period)
    vol_ma_20 = np.full_like(volume, np.nan, dtype=float)
    for i in range(19, len(volume)):
        vol_ma_20[i] = np.mean(volume[i - 19:i + 1])
    
    # Pre-compute 1w ADX(14)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # Align with index 0
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Wilder's smoothing function
    def wheilder_smoothing(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value: simple average
        result[period-1] = np.nansum(arr[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(arr)):
            if not np.isnan(result[i-1]):
                result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    atr_1w = wheilder_smoothing(tr, 14)
    dm_plus_smooth = wheilder_smoothing(dm_plus, 14)
    dm_minus_smooth = wheilder_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.full_like(atr_1w, np.nan, dtype=float)
    di_minus = np.full_like(atr_1w, np.nan, dtype=float)
    for i in range(14, len(atr_1w)):
        if not np.isnan(atr_1w[i]) and atr_1w[i] != 0:
            di_plus[i] = (dm_plus_smooth[i] / atr_1w[i]) * 100
            di_minus[i] = (dm_minus_smooth[i] / atr_1w[i]) * 100
    
    # DX and ADX
    dx = np.full_like(di_plus, np.nan, dtype=float)
    for i in range(14, len(di_plus)):
        if not np.isnan(di_plus[i]) and not np.isnan(di_minus[i]):
            di_sum = di_plus[i] + di_minus[i]
            if di_sum != 0:
                dx[i] = np.abs(di_plus[i] - di_minus[i]) / di_sum * 100
    
    adx_1w = wheilder_smoothing(dx, 14)
    
    # Align HTF indicators to 1d timeframe
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_20[i]) or np.isnan(donchian_low_20[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(adx_1w_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Volume spike condition (1.5x average)
            vol_spike = volume[i] > 1.5 * vol_ma_20[i]
            
            # Long conditions: price > 20-period Donchian high AND volume spike AND 1w strongly trending (ADX > 25)
            if (close[i] > donchian_high_20[i] and vol_spike and adx_1w_aligned[i] > 25):
                position = 1
                signals[i] = 0.25
            # Short conditions: price < 20-period Donchian low AND volume spike AND 1w strongly trending (ADX > 25)
            elif (close[i] < donchian_low_20[i] and vol_spike and adx_1w_aligned[i] > 25):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price returns to 10-period Donchian midpoint or reverses to opposite 20-period Donchian band
            exit_long = (position == 1 and (close[i] < donchian_mid_10[i] or close[i] > donchian_high_20[i] * 1.02))
            exit_short = (position == -1 and (close[i] > donchian_mid_10[i] or close[i] < donchian_low_20[i] * 0.98))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals

def wheilder_smoothing(arr, period):
    result = np.full_like(arr, np.nan, dtype=float)
    if len(arr) < period:
        return result
    # First value: simple average
    result[period-1] = np.nansum(arr[:period])
    # Subsequent values: Wilder's smoothing
    for i in range(period, len(arr)):
        if not np.isnan(result[i-1]):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
    return result