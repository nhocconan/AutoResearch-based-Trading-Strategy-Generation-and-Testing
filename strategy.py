#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot levels (L3/H3) from 1d + volume confirmation + choppiness regime filter
# - Primary signal: Long when price touches 1d Camarilla L3 level with volume spike; Short when price touches H3 level
# - Regime filter: 12h Choppiness Index > 61.8 (ranging market) to avoid false breakouts in strong trends
# - Volume confirmation: 12h volume > 1.5x 20-period median volume
# - Position size: 0.25 (discrete level) for balanced risk/return
# - Target: 12-37 trades/year (50-150 total over 4 years) per 12h strategy guidelines
# - Works in bull/bear: Camarilla levels act as support/resistance in ranging markets; volume confirms participation;
#   chop filter ensures we only trade in ranging regimes where mean reversion at pivots is effective

name = "12h_1d_camarilla_volume_chop_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d indicators for Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels (based on previous day's range)
    # Camarilla: H4 = close + 1.5*(high-low), H3 = close + 1.1*(high-low)
    #            L3 = close - 1.1*(high-low), L4 = close - 1.5*(high-low)
    # We use L3 for longs, H3 for shorts
    range_1d = high_1d - low_1d
    camarilla_h3_1d = close_1d + 1.1 * range_1d
    camarilla_l3_1d = close_1d - 1.1 * range_1d
    
    # Align 1d Camarilla levels to 12h timeframe (completed 1d bar only)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d)
    
    # 12h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h Choppiness Index (14-period) for regime filter
    # Chop = log10(sum(ATR(1)) / (max(high,n) - min(low,n))) / log10(n) * 100
    tr1 = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr1[0] = high[0] - low[0]  # first period TR
    atr1 = pd.Series(tr1).rolling(window=1, min_periods=1).sum().values  # ATR(1) = TR
    sum_atr1_14 = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    denominator = max_high_14 - min_low_14
    chop = np.where(
        denominator == 0,
        50.0,  # neutral when range is zero
        np.log10(sum_atr1_14 / denominator) / np.log10(14) * 100
    )
    chop_regime = chop > 61.8  # ranging market
    
    # 12h volume regime: volume > 1.5x 20-period median volume
    median_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_regime = volume > (1.5 * median_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or
            np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(chop_regime[i]) or
            np.isnan(volume_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses above Camarilla H3 OR chop regime ends (trending market)
            if close[i] >= camarilla_h3_aligned[i] or not chop_regime[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses below Camarilla L3 OR chop regime ends (trending market)
            if close[i] <= camarilla_l3_aligned[i] or not chop_regime[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for price touching Camarilla levels with volume confirmation in ranging regime
            # Long: price touches or crosses above L3 (support) with volume
            if close[i] >= camarilla_l3_aligned[i] and volume_regime[i] and chop_regime[i]:
                position = 1
                signals[i] = 0.25
            # Short: price touches or crosses below H3 (resistance) with volume
            elif close[i] <= camarilla_h3_aligned[i] and volume_regime[i] and chop_regime[i]:
                position = -1
                signals[i] = -0.25
    
    return signals