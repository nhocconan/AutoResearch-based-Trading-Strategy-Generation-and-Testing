#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume spike and 1w chop regime filter
# - Long when price breaks above Camarilla H3 level AND 1d volume > 2.0x 20-period average AND 1w chop < 38.2 (trending market)
# - Short when price breaks below Camarilla L3 level AND 1d volume > 2.0x 20-period average AND 1w chop < 38.2 (trending market)
# - Exit when price returns to Camarilla Pivot point (mean reversion within the pivot structure)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Camarilla pivots work well in ranging markets; volume confirms breakout strength
# - Chop filter ensures we only trade when market is trending (avoid choppy markets where false breakouts occur)
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)

name = "12h_1d_1w_camarilla_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute 12h OHLC
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 12h ATR (14-period) for stoploss
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
    
    # Pre-compute 1w Choppiness Index (14-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w True Range
    tr_1w = np.zeros_like(high_1w)
    tr_1w[0] = high_1w[0] - low_1w[0]
    for i in range(1, len(high_1w)):
        tr_1w[i] = true_range(high_1w[i], low_1w[i], close_1w[i-1])
    
    # Calculate 1w ATR (14-period)
    atr_1w = np.zeros_like(tr_1w)
    atr_1w[13] = np.mean(tr_1w[1:15])
    for i in range(14, len(tr_1w)):
        atr_1w[i] = (atr_1w[i-1] * 13 + tr_1w[i]) / 14
    
    # Calculate 1w Choppiness Index
    hh_1w = highest_high(high_1w, 14)
    ll_1w = lowest_low(low_1w, 14)
    chop_1w = np.zeros_like(close_1w)
    for i in range(13, len(close_1w)):
        if hh_1w[i] > ll_1w[i]:
            log_sum = np.log10(rolling_sum(tr_1w, 14)[i] / (hh_1w[i] - ll_1w[i]))
            chop_1w[i] = 100 * log_sum / np.log10(14)
        else:
            chop_1w[i] = 50.0
    
    chop_regime_1w = chop_1w < 38.2  # Trending market (chop < 38.2)
    
    # Calculate 12h Camarilla Pivot Levels (based on previous 12h bar)
    camarilla_h4 = np.zeros_like(close)
    camarilla_h3 = np.zeros_like(close)
    camarilla_h2 = np.zeros_like(close)
    camarilla_h1 = np.zeros_like(close)
    camarilla_pivot = np.zeros_like(close)
    camarilla_l1 = np.zeros_like(close)
    camarilla_l2 = np.zeros_like(close)
    camarilla_l3 = np.zeros_like(close)
    camarilla_l4 = np.zeros_like(close)
    
    # Calculate pivot levels for each bar using previous bar's OHLC
    for i in range(1, n):
        phigh = high[i-1]
        plow = low[i-1]
        pclose = close[i-1]
        
        pivot = (phigh + plow + pclose) / 3.0
        range_ = phigh - plow
        
        camarilla_pivot[i] = pivot
        camarilla_h1[i] = pivot + (range_ * 1.1 / 12)
        camarilla_h2[i] = pivot + (range_ * 1.1 / 6)
        camarilla_h3[i] = pivot + (range_ * 1.1 / 4)
        camarilla_h4[i] = pivot + (range_ * 1.1 / 2)
        camarilla_l1[i] = pivot - (range_ * 1.1 / 12)
        camarilla_l2[i] = pivot - (range_ * 1.1 / 6)
        camarilla_l3[i] = pivot - (range_ * 1.1 / 4)
        camarilla_l4[i] = pivot - (range_ * 1.1 / 2)
    
    # Align HTF indicators to 12h timeframe
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    chop_regime_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_regime_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(camarilla_pivot[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(chop_regime_1w_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Volume confirmation: current 1d volume > 2.0x 20-period average
            volume_spike = volume[i] > (2.0 * vol_ma_1d_aligned[i])
            
            # Long conditions: price breaks above Camarilla H3 AND volume spike AND trending regime
            if close[i] > camarilla_h3[i] and volume_spike and chop_regime_1w_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below Camarilla L3 AND volume spike AND trending regime
            elif close[i] < camarilla_l3[i] and volume_spike and chop_regime_1w_aligned[i]:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price returns to Camarilla Pivot point
            exit_long = (position == 1 and close[i] <= camarilla_pivot[i])
            exit_short = (position == -1 and close[i] >= camarilla_pivot[i])
            
            # Optional: ATR-based stoploss
            stop_long = (position == 1 and close[i] <= camarilla_h3[i] - 2.0 * atr[i])
            stop_short = (position == -1 and close[i] >= camarilla_l3[i] + 2.0 * atr[i])
            
            if exit_long or exit_short or stop_long or stop_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals

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

def rolling_sum(arr, window):
    result = np.full_like(arr, np.nan, dtype=float)
    for i in range(window - 1, len(arr)):
        result[i] = np.sum(arr[i - window + 1:i + 1])
    return result