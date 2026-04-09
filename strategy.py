#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Williams %R extreme reversal with volume spike and choppiness regime filter
# - Uses 1d HTF for Williams %R(14) to identify oversold/overbought conditions
# - Long when Williams %R < -80 (oversold) with volume > 2.0x 20-period average and chop > 61.8 (ranging market)
# - Short when Williams %R > -20 (overbought) with volume > 2.0x 20-period average and chop > 61.8
# - Fixed position size 0.25 to control drawdown
# - Designed for ranging markets where mean reversion works well (chop > 61.8)
# - Target: 12-30 trades/year on 12h timeframe (50-120 total over 4 years)

name = "12h_1d_williamsr_volume_chop_v1"
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
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Williams %R(14): (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = np.where(
        (highest_high_14 - lowest_low_14) != 0,
        ((highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14)) * -100,
        -50.0  # neutral when range is zero
    )
    
    # Align Williams %R to 12h timeframe (wait for completed 1d bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Pre-compute volume confirmation (20-period average for 12h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute Choppiness Index (14-period) for regime filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = np.where(
        (highest_high_14 - lowest_low_14) != 0,
        -100 * np.log10(atr_sum / (highest_high_14 - lowest_low_14)) / np.log10(14),
        50.0  # neutral when range is zero
    )
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r_aligned[i]) or np.isnan(vol_ma_20[i]) or np.isnan(chop[i]) or
            vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 2.0x average
        volume_confirmed = volume[i] > 2.0 * vol_ma_20[i]
        
        # Regime filter: choppiness > 61.8 indicates ranging market (good for mean reversion)
        ranging_market = chop[i] > 61.8
        
        if position == 1:  # Long position
            # Exit long when Williams %R rises above -50 (mean reversion complete)
            if williams_r_aligned[i] > -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short when Williams %R falls below -50 (mean reversion complete)
            if williams_r_aligned[i] < -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: Williams %R extreme with volume confirmation and ranging market
            if volume_confirmed and ranging_market:
                # Long entry: Williams %R < -80 (oversold)
                if williams_r_aligned[i] < -80:
                    position = 1
                    signals[i] = 0.25
                # Short entry: Williams %R > -20 (overbought)
                elif williams_r_aligned[i] > -20:
                    position = -1
                    signals[i] = -0.25
    
    return signals