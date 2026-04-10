#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d ATR-based volatility filter + chop regime
# - Primary: 4h Donchian breakout (20-period) for trend continuation
# - HTF: 1d ATR ratio (current ATR/20-period MA) > 1.2 for volatility expansion confirmation
# - Regime: 4h choppy market filter (CHOP(14) < 61.8 = avoid breakouts in ranging markets)
# - Long: Price breaks above Donchian upper + volatility expansion + chop < 61.8
# - Short: Price breaks below Donchian lower + volatility expansion + chop < 61.8
# - Exit: Price crosses Donchian middle band or ATR contraction signal
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Works in bull/bear: Donchian adapts to volatility, volatility filter ensures breakout conviction, chop regime avoids whipsaws
# - Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe

name = "4h_1d_donchian_vol_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 25:
        return np.zeros(n)
    
    # Pre-compute 4h data
    close_4h = prices['close'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    
    # Pre-compute 1d data for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 4h Donchian channels (20-period) - use previous bars to avoid look-ahead
    donchian_upper = np.full(len(close_4h), np.nan)
    donchian_lower = np.full(len(close_4h), np.nan)
    donchian_middle = np.full(len(close_4h), np.nan)
    
    for i in range(20, len(close_4h)):
        if not (np.isnan(high_4h[i-20:i]).any() or np.isnan(low_4h[i-20:i]).any()):
            donchian_upper[i] = np.max(high_4h[i-20:i])
            donchian_lower[i] = np.min(low_4h[i-20:i])
            donchian_middle[i] = (donchian_upper[i] + donchian_lower[i]) / 2.0
    
    # Calculate 1d True Range and ATR(20)
    tr_1d = np.full(len(close_1d), np.nan)
    atr_1d = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        if not (np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or np.isnan(close_1d[i-1])):
            tr_1d[i] = max(
                high_1d[i] - low_1d[i],
                abs(high_1d[i] - close_1d[i-1]),
                abs(low_1d[i] - close_1d[i-1])
            )
    
    # Calculate ATR(20) using Wilder's smoothing
    for i in range(20, len(tr_1d)):
        if not np.isnan(tr_1d[i-19:i+1]).any():
            if i == 20:
                atr_1d[i] = np.mean(tr_1d[1:21])  # First ATR is simple average
            else:
                atr_1d[i] = (atr_1d[i-1] * 19 + tr_1d[i]) / 20
    
    # Calculate 1d ATR 20-period moving average for volatility ratio
    atr_ma_20_1d = np.full(len(atr_1d), np.nan)
    for i in range(19, len(atr_1d)):
        if not np.isnan(atr_1d[i-19:i+1]).any():
            atr_ma_20_1d[i] = np.mean(atr_1d[i-19:i+1])
    
    # Calculate 4h Choppiness Index (CHOP) - 14 period
    chop = np.full(len(close_4h), np.nan)
    atr_14 = np.full(len(close_4h), np.nan)
    
    # First calculate True Range and ATR(14) for 4h
    tr_4h = np.full(len(close_4h), np.nan)
    for i in range(1, len(close_4h)):
        if not (np.isnan(high_4h[i]) or np.isnan(low_4h[i]) or np.isnan(close_4h[i-1])):
            tr_4h[i] = max(
                high_4h[i] - low_4h[i],
                abs(high_4h[i] - close_4h[i-1]),
                abs(low_4h[i] - close_4h[i-1])
            )
    
    # Calculate ATR(14) using Wilder's smoothing
    for i in range(14, len(tr_4h)):
        if not np.isnan(tr_4h[i-13:i+1]).any():
            if i == 14:
                atr_14[i] = np.mean(tr_4h[1:15])  # First ATR is simple average
            else:
                atr_14[i] = (atr_14[i-1] * 13 + tr_4h[i]) / 14
    
    # Calculate Choppiness Index: CHOP = 100 * log10(sum(ATR14)/log10(n)) / log10(n)
    for i in range(27, len(close_4h)):  # Need 14 ATR + 14 period for CHOP
        if not np.isnan(atr_14[i-13:i+1]).any():
            sum_atr = np.sum(atr_14[i-13:i+1])
            if sum_atr > 0 and close_4h[i] > 0:
                max_high = np.max(high_4h[i-13:i+1])
                min_low = np.min(low_4h[i-13:i+1])
                if max_high > min_low:
                    chop[i] = 100 * np.log10(sum_atr) / np.log10(14) / np.log10(max_high - min_low)
    
    # Align all HTF indicators to 4h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, prices, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, prices, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, prices, donchian_middle)
    chop_aligned = align_htf_to_ltf(prices, prices, chop)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(28, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or np.isnan(atr_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility expansion: current 1d ATR > 1.2x 20-period MA
        vol_expansion = atr_1d_aligned[i] > 1.2 * atr_ma_20_1d_aligned[i]
        
        # Chop regime filter: only trade when market is trending (CHOP < 61.8)
        trending_regime = chop_aligned[i] < 61.8
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above Donchian upper + vol expansion + trending regime
            if close_4h[i] > donchian_upper_aligned[i] and vol_expansion and trending_regime:
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian lower + vol expansion + trending regime
            elif close_4h[i] < donchian_lower_aligned[i] and vol_expansion and trending_regime:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Price crosses Donchian middle band OR volatility contraction
            if position == 1:  # Long position
                if close_4h[i] < donchian_middle_aligned[i] or atr_1d_aligned[i] < atr_ma_20_1d_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if close_4h[i] > donchian_middle_aligned[i] or atr_1d_aligned[i] < atr_ma_20_1d_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals