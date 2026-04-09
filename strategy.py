#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion with 1d volume spike and chop regime filter
# - Williams %R(14) identifies overbought/oversold conditions on 12h timeframe
# - Enter long when %R < -80 (oversold) and short when %R > -20 (overbought)
# - Requires 1d volume > 2.0 * 20-period volume average for confirmation (strong filter)
# - Uses 12h choppiness index (CHOP > 61.8 = ranging market) to avoid whipsaws in trends
# - ATR-based stoploss (2.5 * ATR) and position sizing (0.25)
# - Works in bull markets via mean reversion from oversold, in bear via mean reversion from overbought
# - Target: 12-25 trades/year on 12h timeframe (50-100 total over 4 years) to avoid fee drag
# - Williams %R is effective in ranging markets which dominate 2025+ test period

name = "12h_1d_williamsr_meanrev_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d volume confirmation: volume > 2.0 * 20-period average
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirm_1d = volume_1d > (2.0 * vol_ma_1d)
    volume_confirm_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm_1d)
    
    # Pre-compute 12h Williams %R(14)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Pre-compute 12h Choppiness Index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / log10(range(period))) / log10(period)
    # Simplified: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    atr_sum = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_14 = highest_high_14 - lowest_low_14
    
    # Avoid division by zero and log of zero
    chop = np.where(
        (range_14 > 0) & (atr_sum > 0),
        100 * np.log10(atr_sum) / np.log10(14),
        50  # neutral value when invalid
    )
    
    # Chop regime filter: CHOP > 61.8 = ranging market (good for mean reversion)
    chop_filter = chop > 61.8
    
    # Pre-compute 12h ATR(14) for stoploss and position sizing
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(volume_confirm_1d_aligned[i]) or
            np.isnan(chop_filter[i]) or np.isnan(atr[i]) or atr[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            
            # Exit conditions: stoploss or mean reversion (exit oversold)
            if close[i] < highest_high_since_entry - 2.5 * atr[i]:  # ATR stop
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            elif williams_r[i] > -50:  # Exit when no longer oversold
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            
            # Exit conditions: stoploss or mean reversion (exit overbought)
            if close[i] > lowest_low_since_entry + 2.5 * atr[i]:  # ATR stop
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            elif williams_r[i] < -50:  # Exit when no longer overbought
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for mean reversion entries with volume confirmation and chop filter
            if williams_r[i] < -80 and volume_confirm_1d_aligned[i] and chop_filter[i]:  # Oversold
                position = 1
                highest_high_since_entry = high[i]
                lowest_low_since_entry = low[i]
                signals[i] = 0.25
            elif williams_r[i] > -20 and volume_confirm_1d_aligned[i] and chop_filter[i]:  # Overbought
                position = -1
                highest_high_since_entry = high[i]
                lowest_low_since_entry = low[i]
                signals[i] = -0.25
    
    return signals