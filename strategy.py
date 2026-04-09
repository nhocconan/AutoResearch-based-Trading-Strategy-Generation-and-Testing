#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Camarilla pivot levels with volume confirmation and choppiness regime filter
# - Uses 1d HTF for Camarilla pivot levels (based on completed daily candles)
# - Long when price touches Camarilla L3 support with volume > 1.5x 20-period average AND chop > 61.8 (range)
# - Short when price touches Camarilla H3 resistance with volume > 1.5x 20-period average AND chop > 61.8 (range)
# - Fixed position size 0.25 to control drawdown
# - Works in bull/bear: Camarilla levels adapt to daily range, volume confirmation filters false signals, chop filter ensures ranging market
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)

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
    # H4 = Pivot + Range * 1.1/2
    # H3 = Pivot + Range * 1.1/4
    # H2 = Pivot + Range * 1.1/6
    # H1 = Pivot + Range * 1.1/12
    # L1 = Pivot - Range * 1.1/12
    # L2 = Pivot - Range * 1.1/6
    # L3 = Pivot - Range * 1.1/4
    # L4 = Pivot - Range * 1.1/2
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    camarilla_h3 = pivot_1d + (range_1d * 1.1 / 4)
    camarilla_l3 = pivot_1d - (range_1d * 1.1 / 4)
    
    # Align Camarilla levels to 12h timeframe (wait for completed 1d bar)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Pre-compute volume confirmation (20-period average for 12h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute Choppiness Index (14-period) for regime filter
    # CHOP = 100 * log10(sum(ATR(1)) / (n * max(HH-LL))) / log10(n)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1 = tr  # ATR(1) is just true range
    
    # Sum of ATR over 14 periods
    sum_atr_14 = pd.Series(atr_1).rolling(window=14, min_periods=14).sum().values
    
    # Max high - min low over 14 periods
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_14 = max_high_14 - min_low_14
    
    # Choppiness Index
    chop = np.zeros(n)
    for i in range(14, n):
        if range_14[i] > 0 and sum_atr_14[i] > 0:
            chop[i] = 100 * np.log10(sum_atr_14[i] / range_14[i]) / np.log10(14)
        else:
            chop[i] = 50  # neutral when invalid
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(chop[i]) or
            vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Regime filter: chop > 61.8 indicates ranging market (good for mean reversion at pivots)
        ranging_market = chop[i] > 61.8
        
        if position == 1:  # Long position
            # Exit long when price moves above L3 (mean reversion complete) or stops ranging
            if close[i] > camarilla_l3_aligned[i] or not ranging_market:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short when price moves below H3 (mean reversion complete) or stops ranging
            if close[i] < camarilla_h3_aligned[i] or not ranging_market:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: Camarilla touch with volume confirmation AND ranging market
            if volume_confirmed and ranging_market:
                # Long entry: price touches or goes below L3 support
                if low[i] <= camarilla_l3_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price touches or goes above H3 resistance
                elif high[i] >= camarilla_h3_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals