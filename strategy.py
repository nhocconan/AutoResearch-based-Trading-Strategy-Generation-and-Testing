#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h volume confirmation and 1w choppiness regime filter
# - Primary: 4h price breaks above/below 20-period Donchian channel for trend continuation
# - Volume filter: 12h volume > 1.3x 20-period volume MA to confirm breakout strength
# - Regime filter: 1w choppiness index < 38.2 indicates trending market (favorable for breakouts)
# - Exit: Price returns to Donchian midpoint (mean reversion within channel) or opposite breakout
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Works in bull/bear: Donchian captures breakouts, volume confirms institutional interest,
#   chop filter avoids false breakouts in ranging markets, effective in both bull and bear trends

name = "4h_12h_1w_donchian_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 20-period Donchian channel for 4h timeframe
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high_20 + lowest_low_20) / 2
    
    # Calculate 12h volume confirmation: volume > 1.3x 20-period volume MA
    volume_ma_20_12h = pd.Series(volume_12h).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_20_12h)
    
    # Calculate 1w choppiness index: CHOP = 100 * log10(sum(ATR(14)) / log10(14 * (HH - LL)))
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = high_1w[0] - low_1w[0]
    tr2[0] = 0
    tr3[0] = 0
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14_sum = pd.Series(true_range).rolling(window=14, min_periods=14).sum().values
    hh14 = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    chop_raw = 100 * np.log10(atr14_sum / np.log10(14 * (hh14 - ll14)))
    chop_raw = np.where((hh14 - ll14) > 0, chop_raw, 50.0)
    chop_1w = chop_raw
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or 
            np.isnan(volume_ma_20_12h_aligned[i]) or np.isnan(chop_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 12h volume > 1.3x 20-period MA
        vol_12h_current = align_htf_to_ltf(prices, df_12h, volume_12h)
        vol_confirm = vol_12h_current[i] > 1.3 * volume_ma_20_12h_aligned[i]
        
        # Regime filter: chop < 38.2 indicates trending market (good for breakouts)
        regime_filter = chop_1w_aligned[i] < 38.2
        
        if position == 0:  # Flat - look for new Donchian breakouts
            # Long entry: Price breaks above upper Donchian + vol confirmation + trending regime
            if close[i] > highest_high_20[i] and vol_confirm and regime_filter:
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below lower Donchian + vol confirmation + trending regime
            elif close[i] < lowest_low_20[i] and vol_confirm and regime_filter:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit when price returns to midpoint or opposite breakout
            # Exit: Price returns to Donchian midpoint or breaks opposite channel
            if position == 1:  # Long position
                if close[i] <= donchian_mid[i] or close[i] < lowest_low_20[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if close[i] >= donchian_mid[i] or close[i] > highest_high_20[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals