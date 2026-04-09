#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with volume confirmation and chop regime filter
# Uses 1d Camarilla levels (H3/L3) for breakout entries, requiring volume spike
# Choppiness Index (CHOP) > 61.8 for ranging market (mean reversion at H3/L3)
# CHOP < 38.2 for trending market (breakout continuation)
# Fixed position size 0.25 to balance return and drawdown
# Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)

name = "4h_1d_camarilla_breakout_chop_volume_v1"
timeframe = "4h"
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
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels (based on previous day)
    # Camarilla: H4 = close + 1.1*(high-low)*1.1/2, H3 = close + 1.1*(high-low)*1.1/4
    # L3 = close - 1.1*(high-low)*1.1/4, L4 = close - 1.1*(high-low)*1.1/2
    # Simplified: use typical levels H3/L3 for breakout
    range_1d = high_1d - low_1d
    camarilla_h3 = close_1d + 1.1 * range_1d * 1.1 / 4
    camarilla_l3 = close_1d - 1.1 * range_1d * 1.1 / 4
    
    # Calculate 1d ATR (14-period) for volatility filtering
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Choppiness Index (14-period) on 1d
    # CHOP = 100 * log10(sum(ATR14) / (max(high,n) - min(low,n))) / log10(n)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max()
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr_14.rolling(window=14, min_periods=14).sum() / (max_high - min_low)) / np.log10(14)
    chop = chop.fillna(50).values  # neutral when undefined
    
    # Align all HTF data to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Pre-compute volume confirmation (20-period average for 4h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(atr_1d_aligned[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(vol_ma_20[i]) or atr_1d_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x average 4h volume
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Chop regime: CHOP > 61.8 = ranging (mean reversion), CHOP < 38.2 = trending
        chop_val = chop_aligned[i]
        is_ranging = chop_val > 61.8
        is_trending = chop_val < 38.2
        
        # Fixed position size
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit on retracement to midpoint of Camarilla H3/L3
            midpoint = (camarilla_h3_aligned[i] + camarilla_l3_aligned[i]) / 2.0
            if close[i] < midpoint:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit on retracement to midpoint of Camarilla H3/L3
            midpoint = (camarilla_h3_aligned[i] + camarilla_l3_aligned[i]) / 2.0
            if close[i] > midpoint:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Camarilla breakout with volume and regime confirmation
            if volume_confirmed:
                if is_ranging:
                    # In ranging market: mean reversion at H3/L3
                    if close[i] > camarilla_h3_aligned[i]:
                        # Sell at H3 resistance (expect reversal down)
                        position = -1
                        signals[i] = -position_size
                    elif close[i] < camarilla_l3_aligned[i]:
                        # Buy at L3 support (expect reversal up)
                        position = 1
                        signals[i] = position_size
                elif is_trending:
                    # In trending market: breakout continuation
                    if close[i] > camarilla_h3_aligned[i]:
                        # Buy breakout above H3
                        position = 1
                        signals[i] = position_size
                    elif close[i] < camarilla_l3_aligned[i]:
                        # Sell breakdown below L3
                        position = -1
                        signals[i] = -position_size
    
    return signals