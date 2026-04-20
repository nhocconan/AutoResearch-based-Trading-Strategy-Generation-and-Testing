#!/usr/bin/env python3
"""
1d_1w_Keltner_Channel_Breakout_Strategy_v1
Concept: Weekly Keltner Channel breakout with volume confirmation.
- Uses weekly EMA20 as center, ATR-based upper/lower bands
- Long when price breaks above upper band with volume > 1.5x average
- Short when price breaks below lower band with volume > 1.5x average
- Exit when price returns to center line (mean reversion)
- Conservative sizing (0.25) to manage drawdown
- Works in bull/bear: Keltner adapts to volatility, volume confirms breakouts
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Keltner_Channel_Breakout_Strategy_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # === Weekly: EMA20 center, ATR(10) for bands ===
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # EMA20 center
    ema20 = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # ATR(10)
    atr10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Keltner Bands
    upper = ema20 + 2 * atr10
    lower = ema20 - 2 * atr10
    
    # Align to daily timeframe
    ema20_aligned = align_htf_to_ltf(prices, df_1w, ema20)
    upper_aligned = align_htf_to_ltf(prices, df_1w, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1w, lower)
    
    # === Daily: Volume confirmation ===
    close = prices['close'].values
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for EMA20
    
    for i in range(start_idx, n):
        # Get values
        ema20_val = ema20_aligned[i]
        upper_val = upper_aligned[i]
        lower_val = lower_aligned[i]
        close_val = close[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema20_val) or np.isnan(upper_val) or np.isnan(lower_val) or 
            np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above upper band with volume confirmation
            breakout_up = close_val > upper_val
            vol_confirm = vol_ratio_val > 1.5
            
            if breakout_up and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower band with volume confirmation
            elif close_val < lower_val and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns to or below center line
            if close_val <= ema20_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns to or above center line
            if close_val >= ema20_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals