#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d volume spike + choppiness regime filter
# - Primary signal: Donchian(20) breakout on 4h close (long: close > upper band, short: close < lower band)
# - Regime filter: 1d choppiness index > 61.8 (range market) for mean reversion, < 38.2 (trend) for trend following
# - Volume confirmation: 4h volume > 1.5 * 20-period median volume (avoid low-participation breakouts)
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 19-50 trades/year (75-200 total over 4 years) per 4h strategy guidelines
# - Works in bull/bear: Choppiness regime adapts strategy - mean revert in range, follow trend in trending markets

name = "4h_1d_donchian_chop_volume_v1"
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
    
    # 1d True Range for choppiness index
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # first bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # 1d ATR(14) and choppiness index
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denom = highest_high_14 - lowest_low_14
    chop = np.where(
        chop_denom == 0,
        50.0,  # neutral when range is zero
        100 * np.log10(sum_tr_14 / chop_denom) / np.log10(14)
    )
    
    # Align 1d choppiness to 4h timeframe (completed 1d bar only)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # 4h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h Donchian(20) bands
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h volume regime: volume > 1.5 * 20-period median volume
    median_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_regime = volume > (1.5 * median_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(chop_aligned[i]) or
            np.isnan(highest_high_20[i]) or
            np.isnan(lowest_low_20[i]) or
            np.isnan(volume_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower band OR choppiness shifts to strong trend
            if close[i] < lowest_low_20[i] or chop_aligned[i] < 38.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper band OR choppiness shifts to strong trend
            if close[i] > highest_high_20[i] or chop_aligned[i] < 38.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with volume confirmation and choppiness filter
            # Long: close breaks above upper band AND volume regime AND chop > 61.8 (range) OR chop < 38.2 (trend) with trend filter
            if close[i] > highest_high_20[i] and volume_regime[i]:
                # In range market (chop > 61.8): mean reversion - wait for pullback before buying breakout
                # In trending market (chop < 38.2): follow trend - buy breakout immediately
                if chop_aligned[i] > 61.8:
                    # Range: mean reversion - only buy if price is near lower band (pullback)
                    if close[i] < (highest_high_20[i] + lowest_low_20[i]) / 2:  # below midpoint
                        position = 1
                        signals[i] = 0.25
                else:
                    # Trend: follow breakout
                    position = 1
                    signals[i] = 0.25
            # Short: close breaks below lower band AND volume regime AND chop > 61.8 (range) OR chop < 38.2 (trend) with trend filter
            elif close[i] < lowest_low_20[i] and volume_regime[i]:
                # In range market (chop > 61.8): mean reversion - wait for pullback before selling breakdown
                # In trending market (chop < 38.2): follow trend - sell breakdown immediately
                if chop_aligned[i] > 61.8:
                    # Range: mean reversion - only sell if price is near upper band (pullback)
                    if close[i] > (highest_high_20[i] + lowest_low_20[i]) / 2:  # above midpoint
                        position = -1
                        signals[i] = -0.25
                else:
                    # Trend: follow breakdown
                    position = -1
                    signals[i] = -0.25
    
    return signals