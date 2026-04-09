#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Camarilla pivot levels + volume spike + choppiness regime filter
# Long when price touches Camarilla L3 level with volume spike in choppy market (mean reversion)
# Short when price touches Camarilla H3 level with volume spike in choppy market
# Uses discrete position sizing 0.25 to target ~20-40 trades/year and minimize fee drag
# Works in bull/bear markets: mean reversion in chop, volume confirmation avoids false signals

name = "4h_1d_camarilla_pivot_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels
    # Camarilla: H4 = close + 1.5*(high-low), H3 = close + 1.1*(high-low), L3 = close - 1.1*(high-low), L4 = close - 1.5*(high-low)
    daily_range = high_1d - low_1d
    camarilla_h3 = close_1d + 1.1 * daily_range
    camarilla_l3 = close_1d - 1.1 * daily_range
    
    # Align 1d Camarilla levels to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Calculate 4h volume spike: current volume > 2.0 * 20-period average volume
    vol_s = pd.Series(volume)
    vol_ma_20 = vol_s.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20
    
    # Calculate 4h choppiness index: CHOP > 61.8 indicates choppy/range market (good for mean reversion)
    def true_range(high, low, prev_close):
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = true_range(high, low, prev_close)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Choppiness Index = 100 * log10(sum(ATR14)/ (n * log10(n))) / log10(n)
    # Simplified: CHOP > 61.8 = choppy, CHOP < 38.2 = trending
    atr_sum_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_14 = max_high_14 - min_low_14
    chop = np.zeros_like(close)
    mask = (range_14 > 0) & (~np.isnan(range_14))
    chop[mask] = 100 * np.log10(atr_sum_14[mask] / range_14[mask]) / np.log10(14)
    chop[~mask] = 50  # neutral value when range is zero
    
    choppy_market = chop > 61.8
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit long if price moves above Camarilla H3 or chop ends
            if close[i] > camarilla_h3_aligned[i] or chop[i] <= 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if price moves below Camarilla L3 or chop ends
            if close[i] < camarilla_l3_aligned[i] or chop[i] <= 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Mean reversion: enter when price touches Camarilla L3/H3 with volume spike in choppy market
            if close[i] <= camarilla_l3_aligned[i] and volume_spike[i] and choppy_market[i]:
                position = 1
                signals[i] = 0.25
            elif close[i] >= camarilla_h3_aligned[i] and volume_spike[i] and choppy_market[i]:
                position = -1
                signals[i] = -0.25
    
    return signals