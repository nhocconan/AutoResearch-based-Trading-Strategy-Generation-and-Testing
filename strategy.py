#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Camarilla pivot levels with volume confirmation and choppiness regime filter
# Camarilla pivots from 1d provide key support/resistance levels aligned with 12h timeframe
# Volume confirmation (current 12h volume > 1.5x 20-period average) filters false breakouts
# Choppiness regime filter: CHOP(14) > 61.8 = ranging market (fade breaks), CHOP < 38.2 = trending (follow breaks)
# Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# Works in bull/bear: price reacts to 1d structure, volume confirms validity, chop filter avoids whipsaws
# Discrete position sizing: 0.0, ±0.30 to balance return and fee drag

name = "12h_1d_camarilla_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 25:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # Resistance levels: R1 = C + Range * 1.1/12, R2 = C + Range * 1.1/6, R3 = C + Range * 1.1/4, R4 = C + Range * 1.1/2
    # Support levels: S1 = C - Range * 1.1/12, S2 = C - Range * 1.1/6, S3 = C - Range * 1.1/4, S4 = C - Range * 1.1/2
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Key levels for trading: R3, R4, S3, S4 (stronger levels)
    camarilla_r3 = close_1d + range_1d * 1.1 / 4.0
    camarilla_r4 = close_1d + range_1d * 1.1 / 2.0
    camarilla_s3 = close_1d - range_1d * 1.1 / 4.0
    camarilla_s4 = close_1d - range_1d * 1.1 / 2.0
    
    # Align Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Pre-compute volume confirmation (20-period average for 12h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute Choppiness Index (14-period) for 12h
    # CHOP = 100 * log10(sum(ATR(1)) / (HHV(high,14) - LLV(low,14))) / log10(14)
    # Where ATR(1) = TR = max(high-low, abs(high-close_prev), abs(low-close_prev))
    tr1 = np.maximum(high - low, np.maximum(np.abs(high - np.concatenate([[close[0]], close[:-1]])), np.abs(low - np.concatenate([[close[0]], close[:-1]]))))
    atr1_sum = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr1_sum / (hh - ll + 1e-10)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x average 12h volume
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Regime filter: CHOP < 38.2 = trending (follow breakouts), CHOP > 61.8 = ranging (fade breaks)
        trending_regime = chop[i] < 38.2
        ranging_regime = chop[i] > 61.8
        
        if position == 1:  # Long position
            # Exit on Camarilla S3 retracement (mean reversion from strong level)
            if close[i] < s3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit on Camarilla R3 retracement (mean reversion from strong level)
            if close[i] > r3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat
            # Breakout trading with volume confirmation and regime filter
            # In trending regime: follow breakouts
            # In ranging regime: fade breaks (mean reversion)
            if volume_confirmed:
                if trending_regime:
                    # Follow breakouts in trending market
                    if close[i] > r4_aligned[i]:
                        position = 1
                        signals[i] = 0.30
                    elif close[i] < s4_aligned[i]:
                        position = -1
                        signals[i] = -0.30
                elif ranging_regime:
                    # Fade breaks in ranging market (mean reversion at extremes)
                    if close[i] < s4_aligned[i]:
                        # Price below strong support -> long (expect bounce)
                        position = 1
                        signals[i] = 0.30
                    elif close[i] > r4_aligned[i]:
                        # Price above strong resistance -> short (expect rejection)
                        position = -1
                        signals[i] = -0.30
    
    return signals