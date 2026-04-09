#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with volume confirmation and chop regime filter
# Uses 1d Camarilla levels (H3/L3) as breakout triggers, volume > 1.5x 20-bar average for confirmation
# Choppiness Index (CHOP) > 61.8 for ranging market (mean reversion at H3/L3), < 38.2 for trending (breakout)
# Long when price breaks above H3 with volume confirmation in trending regime, short when breaks below L3
# Discrete position size 0.25 to target ~20-40 trades/year and minimize fee drag
# Works in bull/bear markets: breakout captures trends, mean reversion in ranges, volume filter reduces false signals

name = "4h_1d_camarilla_breakout_v1"
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
    if len(df_1d) < 1:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels (H3, L3, H4, L4)
    # Camarilla: H4 = close + 1.1*(high-low)*1.1/2, H3 = close + 1.1*(high-low)*1.1/4
    # L3 = close - 1.1*(high-low)*1.1/4, L4 = close - 1.1*(high-low)*1.1/2
    # Using typical close-to-close calculation for simplicity
    hl_range_1d = high_1d - low_1d
    camarilla_h3 = close_1d + 1.1 * hl_range_1d * 1.1 / 4
    camarilla_l3 = close_1d - 1.1 * hl_range_1d * 1.1 / 4
    camarilla_h4 = close_1d + 1.1 * hl_range_1d * 1.1 / 2
    camarilla_l4 = close_1d - 1.1 * hl_range_1d * 1.1 / 2
    
    # Align 1d Camarilla levels to 4h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Calculate 20-period average volume for confirmation
    vol_s = pd.Series(volume)
    avg_vol_20 = vol_s.rolling(window=20, min_periods=20).mean().values
    
    # Calculate Choppiness Index (CHOP) on 1d for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / (n * log10(highest_high - lowest_low)))
    # Simplified: using 14-period True Range and price range
    def true_range(high, low, close_prev):
        tr1 = high - low
        tr2 = np.abs(high - close_prev)
        tr3 = np.abs(low - close_prev)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    close_prev_1d = np.roll(close_1d, 1)
    close_prev_1d[0] = 0
    tr_1d = true_range(high_1d, low_1d, close_prev_1d)
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    highest_high_14_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    price_range_14_1d = highest_high_14_1d - lowest_low_14_1d
    
    # Avoid division by zero
    sum_atr_14 = pd.Series(atr_14_1d).rolling(window=14, min_periods=14).sum().values
    chop_denominator = 14 * np.log10(price_range_14_1d + 1e-10)
    chop_denominator = np.where(chop_denominator == 0, 1e-10, chop_denominator)
    chop = 100 * np.log10(sum_atr_14 + 1e-10) / chop_denominator
    chop = np.where(np.isnan(chop), 50, chop)  # default to neutral
    
    # Align CHOP to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or np.isnan(avg_vol_20[i]) or
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * avg_vol_20[i] if not np.isnan(avg_vol_20[i]) else False
        
        # Regime filter: CHOP > 61.8 = ranging (mean reversion), CHOP < 38.2 = trending (breakout)
        in_trending_regime = chop_aligned[i] < 38.2
        in_ranging_regime = chop_aligned[i] > 61.8
        
        if position == 1:  # Long position
            # Exit long if price falls below L3 (mean reversion in range) or breaks below L4 (stop in trend)
            if in_ranging_regime and close[i] < l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif in_trending_regime and close[i] < l4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if price rises above H3 (mean reversion in range) or breaks above H4 (stop in trend)
            if in_ranging_regime and close[i] > h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif in_trending_regime and close[i] > h4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Breakout strategy: enter on Camarilla breakout with volume confirmation
            if in_trending_regime and volume_confirmed:
                if close[i] > h3_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] < l3_aligned[i]:
                    position = -1
                    signals[i] = -0.25
            # Mean reversion in ranging market: fade extreme levels
            elif in_ranging_regime and volume_confirmed:
                if close[i] > h4_aligned[i]:
                    position = -1  # short at H4
                    signals[i] = -0.25
                elif close[i] < l4_aligned[i]:
                    position = 1   # long at L4
                    signals[i] = 0.25
    
    return signals