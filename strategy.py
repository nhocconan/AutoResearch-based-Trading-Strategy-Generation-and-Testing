#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h volume spike + 1w choppiness regime filter
# - Primary: 4h price breaking above/below 20-period Donchian channels
# - Volume filter: 12h volume > 1.5x 20-period volume MA to confirm breakout strength
# - Regime filter: 1w choppiness index > 61.8 (range market) for mean reversion exits
# - Exit: Price returns to midpoint of Donchian channel or opposite band touch
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Works in bull/bear: Donchian adapts to volatility, volume confirms genuine breakouts,
#   chop filter avoids whipsaws in ranging markets
# - Target: 100-180 total trades over 4 years = 25-45/year for 4h timeframe

name = "4h_12h_1w_donchian_volume_chop_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1w = get_htf_data(prices, '1w')
    if len(df_12h) < 30 or len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 4h Donchian channels (20-period)
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_max_20 + low_min_20) / 2.0
    
    # Calculate 12h volume confirmation: volume > 1.5x 20-period volume MA
    volume_ma_20_12h = pd.Series(volume_12h).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_20_12h)
    
    # Calculate 1w choppiness index: CHOP = 100 * log10(sum(ATR(14)) / log10(N * (HH - LL)))
    # Simplified: CHOP = 100 * log10(sum(True Range over 14 periods) / log10(14 * (Highest High - Lowest Low over 14 periods))
    tr1 = high_1w - low_1w
    tr2 = abs(high_1w - np.roll(close_1w, 1))
    tr3 = abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = high_1w[0] - low_1w[0]  # First bar TR
    tr2[0] = 0
    tr3[0] = 0
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14_sum = pd.Series(true_range).rolling(window=14, min_periods=14).sum().values
    hh14 = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    chop_raw = 100 * np.log10(atr14_sum / np.log10(14 * (hh14 - ll14)))
    chop_raw = np.where((hh14 - ll14) > 0, chop_raw, 50.0)  # Avoid division by zero
    chop_1w = chop_raw
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or np.isnan(donchian_mid[i]) or
            np.isnan(volume_ma_20_12h_aligned[i]) or np.isnan(chop_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 12h volume > 1.5x 20-period MA
        vol_12h_current = align_htf_to_ltf(prices, df_12h, volume_12h)
        vol_confirm = vol_12h_current[i] > 1.5 * volume_ma_20_12h_aligned[i]
        
        # Regime filter: chop > 61.8 indicates ranging market (good for mean reversion)
        regime_filter = chop_1w_aligned[i] > 61.8
        
        if position == 0:  # Flat - look for new entries at Donchian breakouts
            # Long entry: price breaks above upper Donchian + vol confirmation + chop regime
            if close[i] > high_max_20[i] and vol_confirm and regime_filter:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below lower Donchian + vol confirmation + chop regime
            elif close[i] < low_min_20[i] and vol_confirm and regime_filter:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: return to midpoint OR opposite band touch (mean reversion in chop)
            if position == 1:  # Long position
                if close[i] <= donchian_mid[i] or close[i] <= low_min_20[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if close[i] >= donchian_mid[i] or close[i] >= high_max_20[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals