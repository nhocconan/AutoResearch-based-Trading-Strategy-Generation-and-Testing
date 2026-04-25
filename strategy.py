#!/usr/bin/env python3
"""
6h Volume-Weighted RSI + 12h Supertrend Direction + ATR Filter
Hypothesis: Volume-Weighted RSI (VW-RSI) improves on standard RSI by weighting price changes with volume,
reducing false signals during low-participation moves. Combined with 12h Supertrend for trend direction
and ATR-based volatility filter to avoid choppy markets. Works in bull markets (buy VW-RSI<30 in uptrend)
and bear markets (short VW-RSI>70 in downtrend). Targets 12-37 trades/year via strict confluence.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for Supertrend
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h Supertrend (ATR=10, mult=3.0)
    hl2_12h = (df_12h['high'] + df_12h['low']) / 2
    atr_12h = pd.Series(df_12h['high']).rolling(window=10, min_periods=10).max() - \
              pd.Series(df_12h['low']).rolling(window=10, min_periods=10).min()
    atr_12h = pd.Series(atr_12h).ewm(span=10, adjust=False, min_periods=10).mean().values
    upper_12h = hl2_12h + (3.0 * atr_12h)
    lower_12h = hl2_12h - (3.0 * atr_12h)
    supertrend_12h = np.full_like(hl2_12h, np.nan, dtype=float)
    direction_12h = np.ones_like(hl2_12h, dtype=int)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(hl2_12h)):
        if np.isnan(upper_12h[i-1]) or np.isnan(lower_12h[i-1]) or np.isnan(supertrend_12h[i-1]):
            upper_12h[i] = hl2_12h[i] + (3.0 * atr_12h[i])
            lower_12h[i] = hl2_12h[i] - (3.0 * atr_12h[i])
            supertrend_12h[i] = hl2_12h[i]
            direction_12h[i] = 1
        else:
            if close_12h := df_12h['close'].iloc[i]:
                pass
            close_12h = df_12h['close'].iloc[i]
            if supertrend_12h[i-1] == upper_12h[i-1]:
                supertrend_12h[i] = lower_12h[i] if close_12h > lower_12h[i] else upper_12h[i]
                direction_12h[i] = -1 if supertrend_12h[i] == upper_12h[i] else 1
            else:
                supertrend_12h[i] = upper_12h[i] if close_12h < upper_12h[i] else lower_12h[i]
                direction_12h[i] = 1 if supertrend_12h[i] == lower_12h[i] else -1
    
    # Align 12h Supertrend direction to 6h
    direction_12h_aligned = align_htf_to_ltf(prices, df_12h, direction_12h.astype(float))
    
    # Calculate 6h ATR for volatility filter (avoid chop)
    atr_6h = pd.Series(high).rolling(window=14, min_periods=14).max() - \
             pd.Series(low).rolling(window=14, min_periods=14).min()
    atr_6h = pd.Series(atr_6h).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_ma_6h = pd.Series(atr_6h).rolling(window=20, min_periods=20).mean().values
    volatility_filter = atr_6h > (0.7 * atr_ma_6h)  # Only trade when volatility is above 70% of MA
    
    # Calculate Volume-Weighted RSI (6h)
    # VW-RSI = 100 - (100 / (1 + RS)) where RS = Avg Gain_Vol / Avg Loss_Vol
    delta = pd.Series(close).diff().values
    gain = np.where(delta > 0, delta * volume, 0)
    loss = np.where(delta < 0, -delta * volume, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    vw_rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for all calculations
    start_idx = max(20, 14, 10)  # volatility MA, VW-RSI, Supertrend
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(vw_rsi[i]) or np.isnan(direction_12h_aligned[i]) or 
            np.isnan(volatility_filter[i])):
            signals[i] = 0.0
            continue
        
        curr_vw_rsi = vw_rsi[i]
        curr_direction = direction_12h_aligned[i]
        vol_filter = volatility_filter[i]
        
        if position == 0:
            # Look for entry signals
            # Long: VW-RSI < 30 (oversold) AND 12h uptrend AND volatility filter
            long_entry = (curr_vw_rsi < 30) and (curr_direction > 0) and vol_filter
            # Short: VW-RSI > 70 (overbought) AND 12h downtrend AND volatility filter
            short_entry = (curr_vw_rsi > 70) and (curr_direction < 0) and vol_filter
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: VW-RSI > 50 (neutral) OR loss of 12h uptrend
            if (curr_vw_rsi > 50) or (curr_direction <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: VW-RSI < 50 (neutral) OR loss of 12h downtrend
            if (curr_vw_rsi < 50) or (curr_direction >= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_VolWeightedRSI_Supertrend12h_ATRFilter"
timeframe = "6h"
leverage = 1.0