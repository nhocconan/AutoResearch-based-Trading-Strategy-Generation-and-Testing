#!/usr/bin/env python3
"""
4h_Camarilla_R4_S4_Breakout_Volume_TrendFilter_v1
Hypothesis: Price breaking above Camarilla R4 or below S4 from prior 1d session captures strong institutional breakouts with higher follow-through. Combined with volume spike (>2.0x 20-period MA) and 4h EMA50 trend filter (price above/below EMA50). Designed for low trade frequency (~20-40/year) to minimize fee drag and work in both bull (breakout continuation) and bear (breakdown continuation) regimes by requiring strong momentum confirmation and trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Camarilla levels)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === Camarilla levels from prior 1-day session (HLC of previous day) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R4, S4 levels (stronger breakout signals)
    camarilla_r4 = close_1d + (high_1d - low_1d) * 1.1 / 2
    camarilla_s4 = close_1d - (high_1d - low_1d) * 1.1 / 2
    
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # === Volume spike filter (20-period) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    # === 4h EMA50 trend filter ===
    close = prices['close'].values
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(vol_ratio[i]) or np.isnan(ema50[i]) or
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        vol_spike = vol_ratio[i]
        r4 = camarilla_r4_aligned[i]
        s4 = camarilla_s4_aligned[i]
        ema50_val = ema50[i]
        
        if position == 0:
            # Long: price breaks above R4 + volume spike > 2.0 + price above EMA50 (uptrend)
            if price_close > r4 and vol_spike > 2.0 and price_close > ema50_val:
                signals[i] = 0.25
                position = 1
                entry_price = price_close
            # Short: price breaks below S4 + volume spike > 2.0 + price below EMA50 (downtrend)
            elif price_close < s4 and vol_spike > 2.0 and price_close < ema50_val:
                signals[i] = -0.25
                position = -1
                entry_price = price_close
        
        elif position != 0:
            # Stoploss: 2.5 * ATR from entry (wider stop for less whipsaw)
            # Calculate ATR on-the-fly for stoploss (using 14-period ATR)
            if i >= 14:
                high = prices['high'].iloc[i-13:i+1].values
                low = prices['low'].iloc[i-13:i+1].values
                close_arr = prices['close'].iloc[i-13:i+1].values
                
                tr1 = high - low
                tr2 = np.abs(high - np.roll(close_arr, 1))
                tr3 = np.abs(low - np.roll(close_arr, 1))
                tr = np.maximum(tr1, np.maximum(tr2, tr3))
                tr[0] = high[0] - low[0]
                atr_val = np.mean(tr)
            else:
                atr_val = 0.0
            
            if position == 1:
                if price_close < entry_price - 2.5 * atr_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price_close > entry_price + 2.5 * atr_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R4_S4_Breakout_Volume_TrendFilter_v1"
timeframe = "4h"
leverage = 1.0