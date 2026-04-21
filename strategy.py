#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_ATRStop_v1
Hypothesis: On 12h timeframe, price breaking above Camarilla R1 or below S1 from the prior day, with 1-day EMA34 trend filter and ATR-based stoploss, captures momentum breakouts with low trade frequency. Designed for 12h TF to target 50-150 total trades over 4 years (12-37/year) to minimize fee drag and improve generalization to bear markets (2025+).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for EMA trend filter and Camarilla levels)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # === 1-day EMA34 for trend filter ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === Previous day's OHLC for Camarilla levels ===
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    prev_range = prev_high - prev_low
    
    # Camarilla R1 and S1 levels (inner levels for more frequent but still filtered signals)
    camarilla_r1 = prev_close + prev_range * 1.1 / 12
    camarilla_s1 = prev_close - prev_range * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # === ATR for volatility filtering and stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        ema_34 = ema_34_1d_aligned[i]
        r1_level = camarilla_r1_aligned[i]
        s1_level = camarilla_s1_aligned[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: price breaks above R1 + above daily EMA34
            if price_high > r1_level and price_close > ema_34:
                signals[i] = 0.25
                position = 1
                entry_price = price_close
            # Short: price breaks below S1 + below daily EMA34
            elif price_low < s1_level and price_close < ema_34:
                signals[i] = -0.25
                position = -1
                entry_price = price_close
        
        elif position != 0:
            # ATR-based stoploss and trend exit
            if position == 1:
                # Stoploss: 2 * ATR below entry
                stop_price = entry_price - 2.0 * atr_val
                # Exit if price hits stop or trend weakens
                if price_low < stop_price or price_close < ema_34:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Stoploss: 2 * ATR above entry
                stop_price = entry_price + 2.0 * atr_val
                # Exit if price hits stop or trend weakens
                if price_high > stop_price or price_close > ema_34:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_ATRStop_v1"
timeframe = "12h"
leverage = 1.0