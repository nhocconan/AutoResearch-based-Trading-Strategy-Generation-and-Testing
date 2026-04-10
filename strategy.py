#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d ATR volatility filter + chop regime filter
# - Primary: 4h Donchian breakout (20-period) for trend continuation
# - HTF: 1d ATR ratio (ATR(7)/ATR(30) < 0.8) to avoid high volatility false breakouts
# - Regime: 4h choppy market filter (CHOP(14) > 61.8 = avoid breakouts in ranging markets)
# - Long: Price breaks above Donchian upper + ATR filter + chop < 61.8
# - Short: Price breaks below Donchian lower + ATR filter + chop < 61.8
# - Exit: Price crosses opposite Donchian level or ATR filter fails
# - Position sizing: 0.25 (discrete level to balance return and drawdown)
# - Works in bull/bear: Donchian captures trends, ATR filter avoids whipsaws in volatile markets, chop regime avoids false breakouts in ranges
# - Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe

name = "4h_1d_donchian_atr_chop_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:  # Need enough data for ATR(30)
        return np.zeros(n)
    
    # Pre-compute 4h data
    close_4h = prices['close'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 4h Donchian channels (20-period)
    donchian_upper = np.full(len(close_4h), np.nan)
    donchian_lower = np.full(len(close_4h), np.nan)
    
    for i in range(19, len(close_4h)):
        if not (np.isnan(high_4h[i-19:i+1]).any() or np.isnan(low_4h[i-19:i+1]).any()):
            donchian_upper[i] = np.max(high_4h[i-19:i+1])
            donchian_lower[i] = np.min(low_4h[i-19:i+1])
    
    # Calculate 4h Choppiness Index (CHOP) - 14 period
    chop = np.full(len(close_4h), np.nan)
    atr_14 = np.full(len(close_4h), np.nan)
    
    # First calculate True Range and ATR(14)
    tr = np.full(len(close_4h), np.nan)
    for i in range(1, len(close_4h)):
        if not (np.isnan(high_4h[i]) or np.isnan(low_4h[i]) or np.isnan(close_4h[i-1])):
            tr[i] = max(
                high_4h[i] - low_4h[i],
                abs(high_4h[i] - close_4h[i-1]),
                abs(low_4h[i] - close_4h[i-1])
            )
    
    # Calculate ATR(14) using Wilder's smoothing
    for i in range(14, len(tr)):
        if not np.isnan(tr[i-13:i+1]).any():
            if i == 14:
                atr_14[i] = np.mean(tr[1:15])  # First ATR is simple average
            else:
                atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # Calculate Choppiness Index: CHOP = 100 * log10(sum(ATR14)) / log10(14) / log10(max_high - min_low)
    for i in range(27, len(close_4h)):  # Need 14 ATR + 14 period for CHOP
        if not np.isnan(atr_14[i-13:i+1]).any():
            sum_atr = np.sum(atr_14[i-13:i+1])
            if sum_atr > 0:
                max_high = np.max(high_4h[i-13:i+1])
                min_low = np.min(low_4h[i-13:i+1])
                if max_high > min_low:
                    chop[i] = 100 * np.log10(sum_atr) / np.log10(14) / np.log10(max_high - min_low)
    
    # Calculate 1d ATR(7) and ATR(30) for volatility filter
    atr_7_1d = np.full(len(close_1d), np.nan)
    atr_30_1d = np.full(len(close_1d), np.nan)
    
    # Calculate True Range for 1d
    tr_1d = np.full(len(close_1d), np.nan)
    for i in range(1, len(close_1d)):
        if not (np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or np.isnan(close_1d[i-1])):
            tr_1d[i] = max(
                high_1d[i] - low_1d[i],
                abs(high_1d[i] - close_1d[i-1]),
                abs(low_1d[i] - close_1d[i-1])
            )
    
    # Calculate ATR(7) for 1d
    for i in range(7, len(tr_1d)):
        if not np.isnan(tr_1d[i-6:i+1]).any():
            if i == 7:
                atr_7_1d[i] = np.mean(tr_1d[1:8])  # First ATR is simple average
            else:
                atr_7_1d[i] = (atr_7_1d[i-1] * 6 + tr_1d[i]) / 7
    
    # Calculate ATR(30) for 1d
    for i in range(30, len(tr_1d)):
        if not np.isnan(tr_1d[i-29:i+1]).any():
            if i == 30:
                atr_30_1d[i] = np.mean(tr_1d[1:31])  # First ATR is simple average
            else:
                atr_30_1d[i] = (atr_30_1d[i-1] * 29 + tr_1d[i]) / 30
    
    # Calculate ATR ratio: ATR(7)/ATR(30) - low ratio = low volatility
    atr_ratio_1d = np.full(len(close_1d), np.nan)
    for i in range(30, len(atr_7_1d)):
        if not np.isnan(atr_7_1d[i]) and not np.isnan(atr_30_1d[i]) and atr_30_1d[i] > 0:
            atr_ratio_1d[i] = atr_7_1d[i] / atr_30_1d[i]
    
    # Align all HTF indicators to 4h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, prices, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, prices, donchian_lower)
    chop_aligned = align_htf_to_ltf(prices, prices, chop)
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup period for all indicators
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(atr_ratio_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # ATR filter: only trade when volatility is contracting (ATR ratio < 0.8)
        vol_filter = atr_ratio_1d_aligned[i] < 0.8
        
        # Chop regime filter: only trade when market is trending (CHOP < 61.8)
        trending_regime = chop_aligned[i] < 61.8
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above Donchian upper + ATR filter + trending regime
            if close_4h[i] > donchian_upper_aligned[i] and vol_filter and trending_regime:
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian lower + ATR filter + trending regime
            elif close_4h[i] < donchian_lower_aligned[i] and vol_filter and trending_regime:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Price crosses opposite Donchian level OR volatility expands OR chop increases
            if position == 1:  # Long position
                if (close_4h[i] < donchian_lower_aligned[i] or 
                    atr_ratio_1d_aligned[i] >= 0.8 or 
                    chop_aligned[i] >= 61.8):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if (close_4h[i] > donchian_upper_aligned[i] or 
                    atr_ratio_1d_aligned[i] >= 0.8 or 
                    chop_aligned[i] >= 61.8):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals