#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume spike and choppiness regime filter
# - Uses 1d Camarilla pivot levels (H3/L3) as structure for breakout entries
# - Volume confirmation: 4h volume > 2.0x 20-period average to ensure strong breakout
# - Regime filter: 1d Choppiness Index > 61.8 (range) for mean reversion, < 38.2 (trend) for trend following
# - In ranging markets (CHOP > 61.8): fade extreme touches of H3/L3 (mean reversion)
# - In trending markets (CHOP < 38.2): breakout of H3/L3 continues trend
# - ATR(14) trailing stop at 2.0x ATR from extreme for risk control
# - Position size: 0.25 (25% of capital) - discrete level to minimize fee churn
# - Target: ~20-50 trades/year (75-200 total over 4 years) per 4h strategy guidelines
# - Works in both bull/bear: adapts to regime via chop filter, avoids whipsaws in ranging markets

name = "4h_1d_camarilla_breakout_regime_v1"
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
    
    # Calculate 1d ATR(14) for Camarilla pivots
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d[0] = tr_1d[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Camarilla pivot levels (H3, L3) from previous day
    # H3 = close + 1.1*(high-low)/2, L3 = close - 1.1*(high-low)/2
    camarilla_h3 = close_1d + (1.1 * (high_1d - low_1d) / 2)
    camarilla_l3 = close_1d - (1.1 * (high_1d - low_1d) / 2)
    
    # Calculate 1d Choppiness Index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / log10(n)) / log10(n)
    atr_sum = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    n_period = 14
    chop = 100 * np.log10(atr_sum) / np.log10(n_period)
    # Regime: CHOP > 61.8 = ranging (mean revert), CHOP < 38.2 = trending (trend follow)
    chop_regime_ranging = chop > 61.8
    chop_regime_trending = chop < 38.2
    
    # Align 1d indicators to 4h timeframe (completed 1d bar only)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    chop_regime_ranging_aligned = align_htf_to_ltf(prices, df_1d, chop_regime_ranging)
    chop_regime_trending_aligned = align_htf_to_ltf(prices, df_1d, chop_regime_trending)
    
    # 4h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h volume > 2.0x 20-period average (volume confirmation)
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume_20)
    
    # 4h ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(atr[i]) or
            np.isnan(chop_regime_ranging_aligned[i]) or
            np.isnan(chop_regime_trending_aligned[i]) or
            atr[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            if high[i] > highest_since_entry:
                highest_since_entry = high[i]
            
            # Exit conditions: price retraces 2.0x ATR from high
            if low[i] <= highest_since_entry - (2.0 * atr[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            if low[i] < lowest_since_entry:
                lowest_since_entry = low[i]
            
            # Exit conditions: price retraces 2.0x ATR from low
            if high[i] >= lowest_since_entry + (2.0 * atr[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Regime-dependent entry logic
            if chop_regime_ranging_aligned[i]:
                # Ranging market: mean reversion at extreme levels
                # Short when price touches/breaks above H3 (overbought)
                if high[i] >= camarilla_h3_aligned[i] and volume_spike[i]:
                    position = -1
                    highest_since_entry = high[i]
                    lowest_since_entry = high[i]
                    signals[i] = -0.25
                # Long when price touches/breaks below L3 (oversold)
                elif low[i] <= camarilla_l3_aligned[i] and volume_spike[i]:
                    position = 1
                    highest_since_entry = low[i]
                    lowest_since_entry = low[i]
                    signals[i] = 0.25
            elif chop_regime_trending_aligned[i]:
                # Trending market: breakout continuation
                # Long when price breaks above H3 with volume
                if high[i] >= camarilla_h3_aligned[i] and volume_spike[i]:
                    position = 1
                    highest_since_entry = high[i]
                    lowest_since_entry = high[i]
                    signals[i] = 0.25
                # Short when price breaks below L3 with volume
                elif low[i] <= camarilla_l3_aligned[i] and volume_spike[i]:
                    position = -1
                    highest_since_entry = low[i]
                    lowest_since_entry = low[i]
                    signals[i] = -0.25
    
    return signals