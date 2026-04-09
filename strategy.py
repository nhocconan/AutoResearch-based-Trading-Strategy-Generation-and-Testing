#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion with volume spike and choppiness regime filter
# - Williams %R(14) identifies overbought/oversold conditions: < -80 = oversold, > -20 = overbought
# - Long when Williams %R crosses above -80 from below with volume > 2.0x 20-period average
# - Short when Williams %R crosses below -20 from above with volume > 2.0x 20-period average
# - Choppiness regime filter: only trade when CHOP(14) > 61.8 (ranging market) for mean reversion
# - Fixed position size 0.25 to control drawdown
# - Williams %R is effective in ranging markets which dominate 2025+ bear/range conditions
# - Volume spike confirms participation, chop filter avoids trending markets where mean reversion fails
# - Target: 12-30 trades/year on 12h timeframe (48-120 total over 4 years)

name = "12h_1d_williamsr_volume_chop_v2"
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
    
    # Calculate Williams %R on 1d data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14) * -100
    # Handle division by zero when high == low
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # Align Williams %R to 12h timeframe (wait for completed 1d bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate Choppiness Index on 1d data for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(n) * (max(high,n) - min(low,n))))
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr1[0]  # First bar
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    atr_sum = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_hl = max_high - min_low
    
    # Avoid division by zero and log10 of zero
    chop = np.where(
        (range_hl > 0) & (atr_sum > 0),
        100 * np.log10(atr_sum / (np.log10(14) * range_hl)),
        50  # neutral when invalid
    )
    
    # Align Choppiness Index to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Pre-compute volume confirmation (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r_aligned[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(vol_ma_20[i]) or vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 2.0x average
        volume_confirmed = volume[i] > 2.0 * vol_ma_20[i]
        
        # Regime filter: only trade in ranging markets (CHOP > 61.8)
        ranging_market = chop_aligned[i] > 61.8
        
        if position == 1:  # Long position
            # Exit long when Williams %R crosses below -50 (mean reversion complete)
            if williams_r_aligned[i] < -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short when Williams %R crosses above -50 (mean reversion complete)
            if williams_r_aligned[i] > -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: Williams %R extreme + volume confirmation + ranging market
            if volume_confirmed and ranging_market:
                # Long entry: Williams %R crosses above -80 from below (oversold bounce)
                if i > 0 and williams_r_aligned[i-1] <= -80 and williams_r_aligned[i] > -80:
                    position = 1
                    signals[i] = 0.25
                # Short entry: Williams %R crosses below -20 from above (overbought rejection)
                elif i > 0 and williams_r_aligned[i-1] >= -20 and williams_r_aligned[i] < -20:
                    position = -1
                    signals[i] = -0.25
    
    return signals