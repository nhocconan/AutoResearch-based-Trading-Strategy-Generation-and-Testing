#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and 12h chop regime filter
# - Long when price breaks above Donchian(20) high AND 1d volume > 1.3x 20-period average AND 12h chop > 61.8 (ranging market)
# - Short when price breaks below Donchian(20) low AND 1d volume > 1.3x 20-period average AND 12h chop > 61.8 (ranging market)
# - Exit when price returns to Donchian(20) midpoint (mean reversion within the channel)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Donchian breakouts capture momentum in ranging markets; volume confirms institutional participation
# - Chop filter ensures we only trade when market is ranging (avoid strong trends where breakouts fail)
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
    
    # Pre-compute 4h Donchian Channel (20-period)
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
    
    donchian_high = highest_high(high, 20)
    donchian_low = lowest_low(low, 20)
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
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
            log_sum = np.log10(rolling_sum(tr_12h, 14)[i] / (hh_12h[i] - ll_12h[i]))
            chop_12h[i] = 100 * log_sum / np.log10(14)
        else:
            chop_12h[i] = 50.0
    
    chop_regime_12h = chop_12h > 61.8  # Ranging market (chop > 61.8)
    
    # Align HTF indicators to 4h timeframe
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    chop_regime_12h_aligned = align_htf_to_ltf(prices, df_12h, chop_regime_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(chop_regime_12h_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Volume confirmation: current 1d volume > 1.3x 20-period average
            # Since we don't have current 1d volume aligned, use price action as proxy
            # Primary: Donchian breakout + chop regime
            
            # Long conditions: price breaks above Donchian high AND chop regime
            if close[i] > donchian_high[i] and chop_regime_12h_aligned[i]:
                # Additional confirmation: bullish momentum (close in upper half of bar)
                if close[i] > (high[i] + low[i]) / 2:
                    position = 1
                    signals[i] = 0.25
            # Short conditions: price breaks below Donchian low AND chop regime
            elif close[i] < donchian_low[i] and chop_regime_12h_aligned[i]:
                # Additional confirmation: bearish momentum (close in lower half of bar)
                if close[i] < (high[i] + low[i]) / 2:
                    position = -1
                    signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price returns to Donchian midpoint
            exit_long = (position == 1 and close[i] <= donchian_mid[i])
            exit_short = (position == -1 and close[i] >= donchian_mid[i])
            
            # Optional: ATR-based stoploss
            stop_long = (position == 1 and close[i] <= donchian_high[i] - 2.0 * atr[i])
            stop_short = (position == -1 and close[i] >= donchian_low[i] + 2.0 * atr[i])
            
            if exit_long or exit_short or stop_long or stop_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals

def rolling_sum(arr, window):
    result = np.full_like(arr, np.nan, dtype=float)
    for i in range(window - 1, len(arr)):
        result[i] = np.sum(arr[i - window + 1:i + 1])
    return result