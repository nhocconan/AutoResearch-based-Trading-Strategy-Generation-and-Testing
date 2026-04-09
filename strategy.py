#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Williams %R extreme reversal with volume spike and chop filter
# - Williams %R(14) from 12h timeframe identifies overbought/oversold conditions
# - Long when Williams %R < -80 (oversold) with volume > 2.0x 20-period average and chop > 61.8 (ranging market)
# - Short when Williams %R > -20 (overbought) with volume > 2.0x 20-period average and chop > 61.8
# - Fixed position size 0.25 to control drawdown
# - Works in bull/bear: Williams %R captures mean reversion in ranging markets, volume confirms participation
# - Chop filter ensures we only trade in ranging conditions where mean reversion works best
# - Target: 30-60 trades/year on 4h timeframe (120-240 total over 4 years)

name = "4h_12h_williamsr_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r = np.where(
        (highest_high_14 - lowest_low_14) != 0,
        ((highest_high_14 - close_12h) / (highest_high_14 - lowest_low_14)) * -100,
        -50.0  # neutral when range is zero
    )
    
    # Align Williams %R to 4h timeframe (wait for completed 12h bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    
    # Pre-compute volume confirmation (20-period average for 4h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute Choppiness Index (14-period) for 4h
    # Chop = 100 * log10(sum(ATR) / (log10(highest_high - lowest_low) * n))
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = np.where(
        (highest_high_14 - lowest_low_14) > 0,
        100 * np.log10(atr_sum / np.log10(highest_high_14 - lowest_low_14) / 14),
        50.0  # neutral when range is zero
    )
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r_aligned[i]) or np.isnan(vol_ma_20[i]) or np.isnan(chop[i]) or
            vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 2.0x average
        volume_confirmed = volume[i] > 2.0 * vol_ma_20[i]
        
        # Chop filter: only trade in ranging markets (Chop > 61.8)
        chop_filter = chop[i] > 61.8
        
        if position == 1:  # Long position
            # Exit long when Williams %R rises above -50 (mean reversion complete)
            if williams_r_aligned[i] > -50.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short when Williams %R falls below -50 (mean reversion complete)
            if williams_r_aligned[i] < -50.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: Williams %R extreme with volume and chop confirmation
            if volume_confirmed and chop_filter:
                # Long entry: Williams %R < -80 (oversold)
                if williams_r_aligned[i] < -80.0:
                    position = 1
                    signals[i] = 0.25
                # Short entry: Williams %R > -20 (overbought)
                elif williams_r_aligned[i] > -20.0:
                    position = -1
                    signals[i] = -0.25
    
    return signals