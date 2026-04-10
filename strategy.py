#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion + 1d volume spike + chop regime filter
# - Primary: 4h Williams %R(14) < -80 for long, > -20 for short (oversold/overbought)
# - HTF: 1d volume > 2.0x 24-period MA for institutional participation confirmation
# - Regime filter: 4h Choppiness Index (14) > 61.8 = ranging market (mean reversion zone)
# - Long: Williams %R < -80 + volume confirmation + chop ranging
# - Short: Williams %R > -20 + volume confirmation + chop ranging
# - Exit: Williams %R returns to -50 (mean reversion to midpoint)
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Works in bull/bear: Williams %R captures reversals in ranging markets, volume filters weak moves
# - Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe

name = "4h_1d_williamsr_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 4h data
    close_4h = prices['close'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    volume_4h = prices['volume'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 4h Williams %R (14)
    williams_r = np.full(len(close_4h), np.nan)
    for i in range(13, len(close_4h)):
        if not (np.isnan(high_4h[i-13:i+1]).any() or np.isnan(low_4h[i-13:i+1]).any() or np.isnan(close_4h[i])):
            highest_high = np.max(high_4h[i-13:i+1])
            lowest_low = np.min(low_4h[i-13:i+1])
            if highest_high != lowest_low:
                williams_r[i] = -100 * (highest_high - close_4h[i]) / (highest_high - lowest_low)
    
    # Calculate 4h Choppiness Index (14)
    chop = np.full(len(close_4h), np.nan)
    
    # True Range
    tr = np.full(len(close_4h), np.nan)
    for i in range(1, len(close_4h)):
        if not (np.isnan(high_4h[i]) or np.isnan(low_4h[i]) or np.isnan(close_4h[i-1])):
            tr[i] = max(
                high_4h[i] - low_4h[i],
                abs(high_4h[i] - close_4h[i-1]),
                abs(low_4h[i] - close_4h[i-1])
            )
    
    # ATR sum for Chop denominator
    atr_sum = np.full(len(tr), np.nan)
    for i in range(13, len(tr)):
        if not np.isnan(tr[i-13:i+1]).any():
            atr_sum[i] = np.sum(tr[i-13:i+1])
    
    # Choppiness Index
    for i in range(13, len(close_4h)):
        if not (np.isnan(high_4h[i]) or np.isnan(low_4h[i]) or np.isnan(atr_sum[i])):
            highest_high = np.max(high_4h[i-13:i+1])
            lowest_low = np.min(low_4h[i-13:i+1])
            if atr_sum[i] > 0 and (highest_high - lowest_low) > 0:
                chop[i] = 100 * np.log10(atr_sum[i] / (highest_high - lowest_low)) / np.log10(14)
    
    # Calculate 1d volume moving average (24-period)
    volume_ma_24_1d = np.full(len(volume_1d), np.nan)
    for i in range(23, len(volume_1d)):
        if not np.isnan(volume_1d[i-23:i+1]).any():
            volume_ma_24_1d[i] = np.mean(volume_1d[i-23:i+1])
    
    # Align HTF indicators to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, prices, williams_r)
    chop_aligned = align_htf_to_ltf(prices, prices, chop)
    volume_ma_24_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_24_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(volume_ma_24_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 2.0x 24-period MA
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        volume_confirm = volume_1d_aligned[i] > 2.0 * volume_ma_24_1d_aligned[i]
        
        # Chop regime filter: CHOP > 61.8 = ranging market (mean reversion)
        chop_ranging = chop_aligned[i] > 61.8
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Williams %R < -80 (oversold) + volume confirmation + chop ranging
            if williams_r_aligned[i] < -80.0 and volume_confirm and chop_ranging:
                position = 1
                signals[i] = 0.25
            # Short entry: Williams %R > -20 (overbought) + volume confirmation + chop ranging
            elif williams_r_aligned[i] > -20.0 and volume_confirm and chop_ranging:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Williams %R returns to -50 (mean reversion to midpoint)
            if position == 1:  # Long position
                if williams_r_aligned[i] >= -50.0:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if williams_r_aligned[i] <= -50.0:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals