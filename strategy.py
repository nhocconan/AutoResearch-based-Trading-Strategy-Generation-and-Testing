#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Camarilla pivot levels with volume spike and choppiness regime filter
# - Uses 1d HTF for Camarilla pivot levels (based on completed daily candles)
# - Long when price touches Camarilla L3 support with volume > 2.0x average and chop > 61.8 (range)
# - Short when price touches Camarilla H3 resistance with volume > 2.0x average and chop > 61.8 (range)
# - Fixed position size 0.25 to control drawdown
# - Works in bull/bear: Camarilla levels adapt to daily range, volume confirms institutional interest
# - Target: 30-60 trades/year on 12h timeframe (120-240 total over 4 years)

name = "12h_1d_camarilla_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # L3 = C - (Range * 1.1 / 4)
    # H3 = C + (Range * 1.1 / 4)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    camarilla_l3 = close_1d - (range_1d * 1.1 / 4.0)
    camarilla_h3 = close_1d + (range_1d * 1.1 / 4.0)
    
    # Align Camarilla levels to 12h timeframe (wait for completed 1d bar)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    
    # Pre-compute volume confirmation (20-period average for 12h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute Choppiness Index (14-period) for regime filter
    # CHOP = 100 * log10(sum(ATR14) / (max(HH14) - min(LL14))) / log10(14)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * (np.log10(atr14 / (hh14 - ll14 + 1e-10)) / np.log10(14))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_l3_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(chop[i]) or
            vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 2.0x average
        volume_confirmed = volume[i] > 2.0 * vol_ma_20[i]
        
        # Regime filter: chop > 61.8 indicates ranging market (good for mean reversion)
        ranging_market = chop[i] > 61.8
        
        if position == 1:  # Long position
            # Exit long when price moves above L3 (mean reversion complete)
            if close[i] > camarilla_l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short when price moves below H3 (mean reversion complete)
            if close[i] < camarilla_h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: Camarilla touch with volume confirmation and ranging market
            if volume_confirmed and ranging_market:
                # Long entry: price touches or breaks below L3 support
                if low[i] <= camarilla_l3_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price touches or breaks above H3 resistance
                elif high[i] >= camarilla_h3_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals