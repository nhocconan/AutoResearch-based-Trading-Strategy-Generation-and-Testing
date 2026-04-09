#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot levels from 1d + volume spike + choppiness regime filter
# - Uses 1d Camarilla pivot levels (H3, L3, H4, L4) for mean reversion entries
# - Requires volume > 1.8x 20-period average to confirm reversal strength
# - Uses 4h choppiness index (CHOP > 61.8) to ensure ranging market conditions
# - ATR(14) trailing stop at 2.5x ATR from extreme for risk control
# - Position size: 0.25 (25% of capital) - discrete level to minimize fee churn
# - Target: ~20-30 trades/year (80-120 total over 4 years) to stay under fee drag threshold
# - Camarilla pivots work well in ranging markets, volume confirms genuine rejection,
#   chop filter avoids trending markets where mean reversion fails

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
    
    # Pre-compute 1d indicators for Camarilla pivots
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels (based on previous day's range)
    # H4 = close + 1.1 * (high - low) / 2
    # L4 = close - 1.1 * (high - low) / 2
    # H3 = close + 1.1 * (high - low) / 4
    # L3 = close - 1.1 * (high - low) / 4
    # We use previous day's values to avoid look-ahead
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d[0] = high_1d[0]  # first bar uses current values
    prev_low_1d[0] = low_1d[0]
    prev_close_1d[0] = close_1d[0]
    
    camarilla_range = prev_high_1d - prev_low_1d
    h4 = prev_close_1d + 1.1 * camarilla_range / 2
    l4 = prev_close_1d - 1.1 * camarilla_range / 2
    h3 = prev_close_1d + 1.1 * camarilla_range / 4
    l3 = prev_close_1d - 1.1 * camarilla_range / 4
    
    # Align Camarilla levels to 4h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # 4h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h volume > 1.8x 20-period average (volume confirmation)
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * avg_volume_20)
    
    # 4h Choppiness Index (CHOP > 61.8 = ranging market)
    # CHOP = 100 * log10(sum(ATR14) / (max(high,n) - min(low,n))) / log10(n)
    atr_period = 14
    chop_period = 14
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr[0]
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Sum of ATR over chop_period
    sum_atr = pd.Series(atr).rolling(window=chop_period, min_periods=chop_period).sum().values
    
    # Max high and min low over chop_period
    max_high = pd.Series(high).rolling(window=chop_period, min_periods=chop_period).max().values
    min_low = pd.Series(low).rolling(window=chop_period, min_periods=chop_period).min().values
    range_chop = max_high - min_low
    
    # Avoid division by zero
    range_chop = np.where(range_chop == 0, 1e-10, range_chop)
    
    # Choppiness Index
    chop = 100 * (np.log10(sum_atr / range_chop) / np.log10(chop_period))
    chop = np.where(np.isnan(chop), 50.0, chop)  # neutral value when undefined
    chop_regime = chop > 61.8  # ranging market
    
    # 4h ATR(14) for trailing stop
    atr_stop = atr  # reuse ATR calculated above
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(h4_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(h3_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(chop_regime[i]) or
            np.isnan(atr_stop[i]) or atr_stop[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            if high[i] > highest_since_entry:
                highest_since_entry = high[i]
            
            # Exit conditions: price retraces 2.5x ATR from high OR reaches H3 (take profit)
            if low[i] <= highest_since_entry - (2.5 * atr_stop[i]) or high[i] >= h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            if low[i] < lowest_since_entry:
                lowest_since_entry = low[i]
            
            # Exit conditions: price retraces 2.5x ATR from low OR reaches L3 (take profit)
            if high[i] >= lowest_since_entry + (2.5 * atr_stop[i]) or low[i] <= l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for mean reversion at Camarilla levels with volume spike and chop filter
            # Long: price touches or goes below L4 AND closes above L4 AND volume spike AND chop > 61.8
            if (low[i] <= l4_aligned[i] and 
                close[i] > l4_aligned[i] and
                volume_spike[i] and
                chop_regime[i]):
                position = 1
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = 0.25
            # Short: price touches or goes above H4 AND closes below H4 AND volume spike AND chop > 61.8
            elif (high[i] >= h4_aligned[i] and 
                  close[i] < h4_aligned[i] and
                  volume_spike[i] and
                  chop_regime[i]):
                position = -1
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -0.25
    
    return signals