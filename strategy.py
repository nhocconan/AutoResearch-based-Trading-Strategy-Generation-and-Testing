#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot + 1d volume spike + choppiness regime filter
# - Primary signal: Price touches Camarilla H3 (resistance) for short or L3 (support) for long
# - Trend filter: 1d volume > 1.5x 20-period average volume (avoid low-participation signals)
# - Regime filter: 4h Choppiness Index > 61.8 (range-bound market) for mean reversion at pivots
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 20-50 trades/year (75-200 total over 4 years) per 4h strategy guidelines
# - Works in bull/bear: Camarilla pivots adapt to volatility, volume confirms participation,
#   chop filter ensures we only mean revert in ranging markets (avoids trend whipsaw)

name = "4h_1d_camarilla_volume_chop_v1"
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
    
    # Pre-compute 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla levels from previous 1d bar
    # H3 = Close + 1.1*(High-Low)/4, L3 = Close - 1.1*(High-Low)/4
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) / 4
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) / 4
    
    # Align Camarilla levels to 4h timeframe (completed 1d bar only)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # 1d volume regime: volume > 1.5x 20-period average volume
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (1.5 * avg_volume_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    # 4h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h Choppiness Index (14-period)
    # CHOP = 100 * log10(sum(ATR(1)) / (n * log10(highest_high - lowest_low))) / log10(n)
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr1 = np.maximum(tr1, np.absolute(low - np.roll(close, 1)))
    tr1[0] = high[0] - low[0]  # first period
    atr1 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    sum_atr1 = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_denom = highest_high_14 - lowest_low_14
    chop = np.where(
        (chop_denom > 0) & (np.roll(chop_denom, 1) > 0),
        100 * np.log10(sum_atr1 / (14 * np.log10(chop_denom))) / np.log10(14),
        50.0  # neutral when range is zero
    )
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or
            np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(volume_spike_aligned[i]) or
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches midpoint (H3-L3)/2 or chop < 38.2 (trending)
            midpoint = (camarilla_h3_aligned[i] + camarilla_l3_aligned[i]) / 2
            if close[i] >= midpoint or chop[i] < 38.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches midpoint or chop < 38.2 (trending)
            midpoint = (camarilla_h3_aligned[i] + camarilla_l3_aligned[i]) / 2
            if close[i] <= midpoint or chop[i] < 38.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla touch with volume confirmation and chop > 61.8 (range)
            # Long: price <= L3 AND volume spike AND chop > 61.8
            if close[i] <= camarilla_l3_aligned[i] and volume_spike_aligned[i] and chop[i] > 61.8:
                position = 1
                signals[i] = 0.25
            # Short: price >= H3 AND volume spike AND chop > 61.8
            elif close[i] >= camarilla_h3_aligned[i] and volume_spike_aligned[i] and chop[i] > 61.8:
                position = -1
                signals[i] = -0.25
    
    return signals