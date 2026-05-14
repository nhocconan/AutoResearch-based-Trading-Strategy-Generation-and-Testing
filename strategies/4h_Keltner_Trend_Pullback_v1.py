#!/usr/bin/env python3
"""
4h_Keltner_Trend_Pullback_v1
Concept: 4h EMA(200) trend filter with Keltner Channel pullback entries and volume confirmation.
- Long: Price > EMA200 AND pullback to EMA20 (within Keltner lower band) AND volume > 1.5x avg volume
- Short: Price < EMA200 AND pullback to EMA20 (within Keltner upper band) AND volume > 1.5x avg volume
- Exit: Price crosses EMA200 (trend reversal)
- Position sizing: 0.25
- Target: 20-40 trades/year (80-160 total over 4 years)
- Works in bull/bear: EMA200 defines trend, Keltner channels capture pullbacks, volume confirms momentum
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Keltner_Trend_Pullback_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 210:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for volume context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 4h: EMA200 trend filter ===
    close = prices['close'].values
    ema200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # === 4h: EMA20 for pullback target ===
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # === 4h: ATR for Keltner Channel ===
    high = prices['high'].values
    low = prices['low'].values
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # === 4h: Keltner Channels (EMA20 ± 2*ATR) ===
    keltner_upper = ema20 + 2.0 * atr
    keltner_lower = ema20 - 2.0 * atr
    
    # === 4h: Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # === Daily: Volume context (ensure we're not in extremely low volume days) ===
    vol_1d = df_1d['volume'].values
    vol_ma50_1d = pd.Series(vol_1d).rolling(window=50, min_periods=50).mean().values
    vol_ratio_1d = vol_1d / np.where(vol_ma50_1d > 0, vol_ma50_1d, np.nan)
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Get values
        ema200_val = ema200[i]
        ema20_val = ema20[i]
        close_val = close[i]
        keltner_upper_val = keltner_upper[i]
        keltner_lower_val = keltner_lower[i]
        vol_ratio_val = vol_ratio[i]
        vol_ratio_1d_val = vol_ratio_1d_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema200_val) or np.isnan(ema20_val) or np.isnan(keltner_upper_val) or 
            np.isnan(keltner_lower_val) or np.isnan(vol_ratio_val) or np.isnan(vol_ratio_1d_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above EMA200 AND pullback to EMA20 (near lower Keltner) AND volume confirmation
            pullback_long = close_val <= ema20_val * 1.02 and close_val >= keltner_lower_val
            vol_confirm = vol_ratio_val > 1.5 and vol_ratio_1d_val > 0.8  # Not extremely low volume day
            
            if close_val > ema200_val and pullback_long and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Price below EMA200 AND pullback to EMA20 (near upper Keltner) AND volume confirmation
            elif close_val < ema200_val and close_val >= ema20_val * 0.98 and close_val <= keltner_upper_val and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses below EMA200 (trend change)
            if close_val < ema200_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses above EMA200 (trend change)
            if close_val > ema200_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals