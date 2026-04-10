#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with 1d chop regime filter
# - Long when Williams %R(14) crosses above -80 (oversold) AND 1d chop > 61.8 (ranging market)
# - Short when Williams %R(14) crosses below -20 (overbought) AND 1d chop > 61.8 (ranging market)
# - Exit when Williams %R returns to -50 (mean reversion)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Williams %R works well in ranging markets which are common in bear/range regimes like 2025+
# - Chop filter ensures we only trade when market is ranging (avoid trending markets where mean reversion fails)
# - Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)

name = "4h_1d_williamsr_chop_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute 4h OHLC
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Pre-compute 4h Williams %R (14-period)
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
    
    hh = highest_high(high, 14)
    ll = lowest_low(low, 14)
    williams_r = np.zeros_like(close)
    for i in range(13, len(close)):
        if hh[i] != ll[i]:  # Avoid division by zero
            williams_r[i] = -100 * (hh[i] - close[i]) / (hh[i] - ll[i])
        else:
            williams_r[i] = -50.0  # Neutral when range is zero
    
    # Pre-compute 4h Choppiness Index (14-period)
    def true_range(h, l, c_prev):
        tr1 = h - l
        tr2 = np.abs(h - c_prev)
        tr3 = np.abs(l - c_prev)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    def rolling_sum(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.sum(arr[i - window + 1:i + 1])
        return result
    
    # Calculate True Range
    tr = np.zeros_like(high)
    tr[0] = high[0] - low[0]  # First bar
    for i in range(1, len(high)):
        tr[i] = true_range(high[i], low[i], close[i-1])
    
    # Calculate ATR (14-period) using Wilder's smoothing
    atr = np.zeros_like(tr)
    atr[13] = np.mean(tr[1:15])  # First ATR value
    for i in range(14, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Calculate Choppiness Index
    hh_chop = highest_high(high, 14)
    ll_chop = lowest_low(low, 14)
    chop = np.zeros_like(close)
    for i in range(13, len(close)):
        if hh_chop[i] > ll_chop[i]:
            log_sum = np.log10(rolling_sum(tr, 14)[i] / (hh_chop[i] - ll_chop[i]))
            chop[i] = 100 * log_sum / np.log10(14)
        else:
            chop[i] = 50.0
    
    chop_regime = chop > 61.8  # Ranging market (chop > 61.8)
    
    # Pre-compute 1d Choppiness Index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d True Range
    tr_1d = np.zeros_like(high_1d)
    tr_1d[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(high_1d)):
        tr_1d[i] = true_range(high_1d[i], low_1d[i], close_1d[i-1])
    
    # Calculate 1d ATR (14-period)
    atr_1d = np.zeros_like(tr_1d)
    atr_1d[13] = np.mean(tr_1d[1:15])
    for i in range(14, len(tr_1d)):
        atr_1d[i] = (atr_1d[i-1] * 13 + tr_1d[i]) / 14
    
    # Calculate 1d Choppiness Index
    hh_1d = highest_high(high_1d, 14)
    ll_1d = lowest_low(low_1d, 14)
    chop_1d = np.zeros_like(close_1d)
    for i in range(13, len(close_1d)):
        if hh_1d[i] > ll_1d[i]:
            log_sum = np.log10(rolling_sum(tr_1d, 14)[i] / (hh_1d[i] - ll_1d[i]))
            chop_1d[i] = 100 * log_sum / np.log10(14)
        else:
            chop_1d[i] = 50.0
    
    chop_regime_1d = chop_1d > 61.8  # 1d ranging market
    
    # Align HTF indicators to 4h timeframe
    chop_regime_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_regime_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(chop_regime_1d_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: Williams %R crosses above -80 (oversold) AND 1d chop regime (ranging)
            if i > 0 and williams_r[i-1] <= -80 and williams_r[i] > -80 and chop_regime_1d_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short conditions: Williams %R crosses below -20 (overbought) AND 1d chop regime (ranging)
            elif i > 0 and williams_r[i-1] >= -20 and williams_r[i] < -20 and chop_regime_1d_aligned[i]:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: Williams %R returns to -50 (mean reversion)
            exit_long = (position == 1 and williams_r[i] >= -50)
            exit_short = (position == -1 and williams_r[i] <= -50)
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals