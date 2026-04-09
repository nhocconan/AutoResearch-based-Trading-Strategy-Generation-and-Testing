#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with 12h volume spike and chop regime filter
# - Uses 4h Williams %R(14) for oversold/overbought signals (long when %R < -80, short when %R > -20)
# - Confirms with 12h volume > 2.0x 20-period average (strong institutional participation)
# - Filters by 12h choppiness index: trade only when CHOP > 61.8 (range) for mean reversion
# - Exits when Williams %R returns to -50 level or ATR-based stoploss (2.0x ATR)
# - Position size: 0.25 (25% of capital) to limit drawdown in volatile markets
# - Target: 15-30 trades/year on 4h timeframe (60-120 total over 4 years) to minimize fee drag
# - Works in bull markets (mean reversion in uptrend) and bear markets (mean reversion in downtrend)
# - Williams %R identifies exhaustion points that work across market regimes

name = "4h_12h_williamsr_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Pre-compute HTF indicators
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # 12h True Range for ATR and chop
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # 12h ATR(14) for stoploss
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 12h Volume > 2.0x 20-period average (stricter for fewer trades)
    avg_volume_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_12h > (2.0 * avg_volume_20)
    
    # 12h Choppiness Index(14)
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_14 = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_14 = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    chop_denom = np.where((highest_14 - lowest_14) > 0, highest_14 - lowest_14, 1e-10)
    chop = 100 * np.log10(sum_tr_14 / chop_denom) / np.log10(14)
    chop_range = chop > 61.8  # range-bound market (good for mean reversion)
    
    # Align all 12h indicators to 4h
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    volume_spike_aligned = align_htf_to_ltf(prices, df_12h, volume_spike.astype(float))
    chop_range_aligned = align_htf_to_ltf(prices, df_12h, chop_range.astype(float))
    
    # 4h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 4h Williams %R(14)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14 + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    atr_stop = 0.0
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(atr_12h_aligned[i]) or
            np.isnan(volume_spike_aligned[i]) or np.isnan(chop_range_aligned[i]) or
            atr_12h_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions: Williams %R returns to -50 or ATR stoploss
            if williams_r[i] >= -50:  # Return to midpoint
                position = 0
                signals[i] = 0.0
            elif low[i] <= entry_price - (2.0 * atr_stop):  # ATR stoploss
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: Williams %R returns to -50 or ATR stoploss
            if williams_r[i] <= -50:  # Return to midpoint
                position = 0
                signals[i] = 0.0
            elif high[i] >= entry_price + (2.0 * atr_stop):  # ATR stoploss
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Williams %R extremes with volume confirmation and range regime
            if (williams_r[i] <= -80 and  # Oversold
                volume_spike_aligned[i] and         # Volume confirmation
                chop_range_aligned[i]):             # Range-bound market
                position = 1
                entry_price = low[i]
                atr_stop = atr_12h_aligned[i]
                signals[i] = 0.25
            elif (williams_r[i] >= -20 and    # Overbought
                  volume_spike_aligned[i] and         # Volume confirmation
                  chop_range_aligned[i]):             # Range-bound market
                position = -1
                entry_price = high[i]
                atr_stop = atr_12h_aligned[i]
                signals[i] = -0.25
    
    return signals