#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout + 1d volume spike + 4h choppiness regime filter
# - Uses Camarilla pivot levels (L3, L4, H3, H4) from 1d for breakout entries
# - Uses 1d volume spike (volume > 1.5x 20-period average) to confirm institutional interest
# - Uses 4h choppiness index (CHOP > 61.8 = ranging, CHOP < 38.2 = trending) to filter regime
# - Long when price breaks above H3/H4 with volume spike and CHOP < 38.2 (trending)
# - Short when price breaks below L3/L4 with volume spike and CHOP < 38.2 (trending)
# - Fixed position size 0.25 to control drawdown
# - ATR-based stoploss: exit when price moves 2.0x ATR against position
# - Works in both bull and bear markets by only trading in trending regimes with volume confirmation
# - Camarilla pivots provide mathematically derived support/resistance levels
# - Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)

name = "4h_1d_camarilla_breakout_volume_chop_v1"
timeframe = "4h"
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
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume spike (volume > 1.5x 20-period average)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (1.5 * vol_ma_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    # Calculate 1d Camarilla pivot levels
    # Camarilla formulas: 
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.125 * (high - low)
    # L3 = close - 1.125 * (high - low)
    # L4 = close - 1.5 * (high - low)
    camarilla_high = high_1d - low_1d
    h4 = close_1d + 1.5 * camarilla_high
    h3 = close_1d + 1.125 * camarilla_high
    l3 = close_1d - 1.125 * camarilla_high
    l4 = close_1d - 1.5 * camarilla_high
    
    # Align Camarilla levels to 4h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # Calculate 4h choppiness index (CHOP)
    # CHOP = 100 * log10(sum(ATR1) / (ATR14)) / log10(14)
    # Where ATR1 = True Range of current bar
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar
    
    atr1_sum = pd.Series(tr).rolling(window=1, min_periods=1).sum().values  # Just TR
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Avoid division by zero and log of zero
    chop_raw = np.where((atr14 > 0) & (atr1_sum > 0), 
                        100 * np.log10(atr1_sum / atr14) / np.log10(14), 
                        50)  # Default to neutral when invalid
    chop = pd.Series(chop_raw).rolling(window=14, min_periods=14).mean().values
    
    # Regime filter: CHOP < 38.2 = trending (good for breakouts)
    trending_regime = chop < 38.2
    
    # Pre-compute ATR (14-period) for stoploss
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(h4_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(volume_spike_aligned[i]) or np.isnan(trending_regime[i]) or
            np.isnan(atr[i]) or atr[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            
            # ATR-based trailing stop: exit if price drops 2.0x ATR from highest high
            if close[i] < highest_high_since_entry - 2.0 * atr[i]:
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            
            # ATR-based trailing stop: exit if price rises 2.0x ATR from lowest low
            if close[i] > lowest_low_since_entry + 2.0 * atr[i]:
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_spike_aligned[i] and trending_regime[i]:
                # Long entry: price breaks above H3 or H4
                if close[i] > h3_aligned[i] or close[i] > h4_aligned[i]:
                    position = 1
                    highest_high_since_entry = high[i]
                    lowest_low_since_entry = low[i]
                    signals[i] = 0.25
                # Short entry: price breaks below L3 or L4
                elif close[i] < l3_aligned[i] or close[i] < l4_aligned[i]:
                    position = -1
                    highest_high_since_entry = high[i]
                    lowest_low_since_entry = low[i]
                    signals[i] = -0.25
    
    return signals