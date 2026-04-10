#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 1d chop regime filter
# - Long when price breaks above Donchian(20) high AND 4h volume > 1.5x 20-period average AND 1d chop > 61.8 (ranging market mean reversion)
# - Short when price breaks below Donchian(20) low AND 4h volume > 1.5x 20-period average AND 1d chop > 61.8
# - Exit when price crosses Donchian(20) midpoint (mean reversion in ranging markets)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Donchian provides adaptive structure; volume confirms breakout strength
# - Chop filter ensures we only mean revert in ranging markets (avoid trending whipsaws)
# - Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years)

name = "4h_1d_donchian_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 4h OHLC
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 4h Donchian(20)
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
    
    donchian_high = rolling_max(high, 20)
    donchian_low = rolling_min(low, 20)
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Pre-compute 4h volume average (20-period)
    def rolling_mean(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.mean(arr[i - window + 1:i + 1])
        return result
    
    vol_ma_4h = rolling_mean(volume, 20)
    
    # Pre-compute 1d Chop Index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    def true_range(h, l, c_prev):
        tr1 = h - l
        tr2 = np.abs(h - c_prev)
        tr3 = np.abs(l - c_prev)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    tr_1d = np.zeros_like(high_1d)
    tr_1d[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(high_1d)):
        tr_1d[i] = true_range(high_1d[i], low_1d[i], close_1d[i-1])
    
    atr_1d = np.zeros_like(tr_1d)
    for i in range(len(tr_1d)):
        if i < 14:
            atr_1d[i] = np.nan
        else:
            atr_1d[i] = np.mean(tr_1d[max(0, i-13):i+1])
    
    # Chop Index = 100 * log10(sum(ATR14) / (max(high) - min(low))) / log10(14)
    chop_1d = np.full_like(close_1d, np.nan, dtype=float)
    for i in range(13, len(close_1d)):
        if np.isnan(atr_1d[i]) or np.isnan(high_1d[i]) or np.isnan(low_1d[i]):
            chop_1d[i] = np.nan
            continue
        sum_atr = np.sum(atr_1d[i-13:i+1])
        max_high = np.max(high_1d[i-13:i+1])
        min_low = np.min(low_1d[i-13:i+1])
        if max_high == min_low:
            chop_1d[i] = 50.0
        else:
            chop_1d[i] = 100 * np.log10(sum_atr / (max_high - min_low)) / np.log10(14)
    
    # Align HTF indicators to 4h timeframe
    vol_ma_4h_aligned = align_htf_to_ltf(prices, prices, vol_ma_4h)  # 4h data already aligned
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(vol_ma_4h_aligned[i]) or 
            np.isnan(chop_1d_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Volume spike condition
            vol_spike = volume[i] > 1.5 * vol_ma_4h_aligned[i]
            
            # Chop regime condition (only trade in ranging markets)
            chop_regime = chop_1d_aligned[i] > 61.8
            
            # Long conditions: Donchian high breakout AND volume spike AND chop regime
            if close[i] > donchian_high[i] and vol_spike and chop_regime:
                position = 1
                signals[i] = 0.25
            # Short conditions: Donchian low breakdown AND volume spike AND chop regime
            elif close[i] < donchian_low[i] and vol_spike and chop_regime:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price crosses Donchian midpoint (mean reversion)
            exit_long = (position == 1 and close[i] < donchian_mid[i])
            exit_short = (position == -1 and close[i] > donchian_mid[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals