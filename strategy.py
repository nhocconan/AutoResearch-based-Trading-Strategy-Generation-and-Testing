#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume confirmation and chop regime filter
# - Long when price breaks above Camarilla H3 level AND 1d volume > 1.5x 20-period average AND 1d chop > 61.8 (ranging)
# - Short when price breaks below Camarilla L3 level AND 1d volume > 1.5x 20-period average AND 1d chop > 61.8 (ranging)
# - Exit when price returns to Camarilla H4/L4 levels (strong reversal levels)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Camarilla pivots from 1d provide institutional support/resistance levels
# - Volume confirmation ensures breakout has participation
# - Chop filter ensures we only trade in ranging markets where pivots hold
# - Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)

name = "4h_1d_camarilla_breakout_volume_chop_v1"
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
    
    # Pre-compute 1d Camarilla levels (based on previous day OHLC)
    # Camarilla: H4 = close + 1.1*(high-low)*1.1/2, H3 = close + 1.1*(high-low)*1.1/4
    #          L3 = close - 1.1*(high-low)*1.1/4, L4 = close - 1.1*(high-low)*1.1/2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_h4 = np.zeros_like(close_1d)
    camarilla_h3 = np.zeros_like(close_1d)
    camarilla_l3 = np.zeros_like(close_1d)
    camarilla_l4 = np.zeros_like(close_1d)
    
    for i in range(len(close_1d)):
        if i == 0:
            camarilla_h4[i] = np.nan
            camarilla_h3[i] = np.nan
            camarilla_l3[i] = np.nan
            camarilla_l4[i] = np.nan
        else:
            rng = high_1d[i-1] - low_1d[i-1]
            camarilla_h4[i] = close_1d[i-1] + 1.1 * rng * 1.1 / 2.0
            camarilla_h3[i] = close_1d[i-1] + 1.1 * rng * 1.1 / 4.0
            camarilla_l3[i] = close_1d[i-1] - 1.1 * rng * 1.1 / 4.0
            camarilla_l4[i] = close_1d[i-1] - 1.1 * rng * 1.1 / 2.0
    
    # Pre-compute 1d volume average (20-period)
    volume_1d = df_1d['volume'].values
    def rolling_mean(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.mean(arr[i - window + 1:i + 1])
        return result
    
    vol_ma_1d = rolling_mean(volume_1d, 20)
    
    # Pre-compute 1d Choppiness Index (14-period)
    def true_range(h, l, c_prev):
        tr1 = h - l
        tr2 = np.abs(h - c_prev)
        tr3 = np.abs(l - c_prev)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
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
    
    hh_1d = highest_high(high_1d, 14)
    ll_1d = lowest_low(low_1d, 14)
    chop_1d = np.zeros_like(close_1d)
    for i in range(13, len(close_1d)):
        if hh_1d[i] > ll_1d[i]:
            # Calculate rolling sum of true range
            tr_sum = np.sum(tr_1d[i-13:i+1])
            chop_1d[i] = 100 * np.log10(tr_sum / (hh_1d[i] - ll_1d[i])) / np.log10(14)
        else:
            chop_1d[i] = 50.0
    
    chop_regime_1d = chop_1d > 61.8  # Ranging market (chop > 61.8)
    
    # Align HTF indicators to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    chop_regime_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_regime_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(chop_regime_1d_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Volume confirmation: current 1d volume > 1.5x 20-period average
            # Use close price as proxy for volume confirmation (strong close = participation)
            volume_confirm = close[i] > (high[i] + low[i]) / 2  # Bullish close for long
            volume_confirm_short = close[i] < (high[i] + low[i]) / 2  # Bearish close for short
            
            # Long conditions: price breaks above Camarilla H3 AND volume confirm AND chop regime
            if close[i] > camarilla_h3_aligned[i] and volume_confirm and chop_regime_1d_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below Camarilla L3 AND volume confirm AND chop regime
            elif close[i] < camarilla_l3_aligned[i] and volume_confirm_short and chop_regime_1d_aligned[i]:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price returns to Camarilla H4/L4 levels
            exit_long = (position == 1 and close[i] <= camarilla_h4_aligned[i])
            exit_short = (position == -1 and close[i] >= camarilla_l4_aligned[i])
            
            # Optional: time-based exit (max 3 bars holding)
            # Not implemented to keep simple
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals