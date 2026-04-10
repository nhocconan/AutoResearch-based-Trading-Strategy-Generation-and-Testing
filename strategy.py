#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d volume confirmation and 12h chop regime filter
# - Long when price breaks above Donchian(20) high AND 1d volume > 1.5x 20-day average AND 12h chop < 38.2 (trending market)
# - Short when price breaks below Donchian(20) low AND 1d volume > 1.5x 20-day average AND 12h chop < 38.2 (trending market)
# - Exit when price crosses Donchian(10) midline OR ATR-based stoploss (2x ATR)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Donchian captures structural breaks; volume confirms institutional participation; chop filter avoids whipsaws in ranging markets
# - Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years)

name = "4h_1d_12h_donchian_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_12h = get_htf_data(prices, '12h')
    if len(df_1d) < 30 or len(df_12h) < 30:
        return np.zeros(n)
    
    # Pre-compute 4h OHLC
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 4h Donchian channels (20-period)
    def highest_high(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.max(arr[i - window + 1:i + 1])
        return result
    
    def lowest_low(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.min(arr[i - window + 1:i + 1])
        return result
    
    hh_20 = highest_high(high, 20)
    ll_20 = lowest_low(low, 20)
    hh_10 = highest_high(high, 10)
    ll_10 = lowest_low(low, 10)
    
    # Pre-compute 4h ATR (14-period) for stoploss
    def true_range(h, l, c_prev):
        tr1 = h - l
        tr2 = np.abs(h - c_prev)
        tr3 = np.abs(l - c_prev)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    tr = np.zeros_like(high)
    tr[0] = high[0] - low[0]
    for i in range(1, len(high)):
        tr[i] = true_range(high[i], low[i], close[i-1])
    
    atr = np.zeros_like(tr)
    atr[13] = np.mean(tr[1:15])  # First ATR value
    for i in range(14, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Pre-compute 1d volume average (20-period)
    volume_1d = df_1d['volume'].values
    def rolling_mean(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.mean(arr[i - window + 1:i + 1])
        return result
    
    vol_ma_1d = rolling_mean(volume_1d, 20)
    
    # Pre-compute 12h Choppiness Index (14-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h True Range
    tr_12h = np.zeros_like(high_12h)
    tr_12h[0] = high_12h[0] - low_12h[0]
    for i in range(1, len(high_12h)):
        tr_12h[i] = true_range(high_12h[i], low_12h[i], close_12h[i-1])
    
    # Calculate 12h ATR (14-period)
    atr_12h = np.zeros_like(tr_12h)
    atr_12h[13] = np.mean(tr_12h[1:15])
    for i in range(14, len(tr_12h)):
        atr_12h[i] = (atr_12h[i-1] * 13 + tr_12h[i]) / 14
    
    # Calculate 12h Choppiness Index
    hh_12h = highest_high(high_12h, 14)
    ll_12h = lowest_low(low_12h, 14)
    chop_12h = np.zeros_like(close_12h)
    for i in range(13, len(close_12h)):
        if hh_12h[i] > ll_12h[i]:
            # Calculate rolling sum of True Range
            tr_sum = np.sum(tr_12h[i-13:i+1])
            chop_12h[i] = 100 * np.log10(tr_sum / (hh_12h[i] - ll_12h[i])) / np.log10(14)
        else:
            chop_12h[i] = 50.0
    
    chop_regime_12h = chop_12h < 38.2  # Trending market (chop < 38.2)
    
    # Align HTF indicators to 4h timeframe
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    chop_regime_12h_aligned = align_htf_to_ltf(prices, df_12h, chop_regime_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(hh_20[i]) or np.isnan(ll_20[i]) or np.isnan(hh_10[i]) or 
            np.isnan(ll_10[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(chop_regime_12h_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: Donchian breakout up AND volume spike AND trending regime
            if close[i] > hh_20[i-1] and volume[i] > 1.5 * vol_ma_1d_aligned[i] and chop_regime_12h_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short conditions: Donchian breakout down AND volume spike AND trending regime
            elif close[i] < ll_20[i-1] and volume[i] > 1.5 * vol_ma_1d_aligned[i] and chop_regime_12h_aligned[i]:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: Donchian(10) midline cross OR ATR-based stoploss
            exit_long = (position == 1 and close[i] < (hh_10[i-1] + ll_10[i-1]) / 2)
            exit_short = (position == -1 and close[i] > (hh_10[i-1] + ll_10[i-1]) / 2)
            
            # ATR-based stoploss
            stop_long = (position == 1 and close[i] <= high[i] - 2.0 * atr[i])
            stop_short = (position == -1 and close[i] >= low[i] + 2.0 * atr[i])
            
            if exit_long or exit_short or stop_long or stop_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals

def rolling_mean(arr, window):
    result = np.full_like(arr, np.nan, dtype=float)
    for i in range(window - 1, len(arr)):
        result[i] = np.mean(arr[i - window + 1:i + 1])
    return result