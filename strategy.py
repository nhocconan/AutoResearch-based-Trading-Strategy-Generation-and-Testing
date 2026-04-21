#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_ATRStop_v1
Hypothesis: On 4h timeframe, price breaking above Camarilla R3 or below S3 indicates strong momentum breakout. Uses 1-day EMA34 for trend filter and ATR-based stoploss. Designed for low trade frequency (target: 20-50/year) to minimize fee drag and work in both bull and bear markets via trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for EMA trend filter)
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
    
    # Camarilla R3 and S3 levels
    camarilla_r3 = prev_close + prev_range * 1.1 / 4
    camarilla_s3 = prev_close - prev_range * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
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
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        ema_34 = ema_34_1d_aligned[i]
        r3_level = camarilla_r3_aligned[i]
        s3_level = camarilla_s3_aligned[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: price breaks above R3 + above daily EMA34
            if price_high > r3_level and price_close > ema_34:
                signals[i] = 0.30
                position = 1
                entry_price = price_close
            # Short: price breaks below S3 + below daily EMA34
            elif price_low < s3_level and price_close < ema_34:
                signals[i] = -0.30
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
                    signals[i] = 0.30
            else:  # position == -1
                # Stoploss: 2 * ATR above entry
                stop_price = entry_price + 2.0 * atr_val
                # Exit if price hits stop or trend weakens
                if price_high > stop_price or price_close > ema_34:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.30
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_ATRStop_v1"
timeframe = "4h"
leverage = 1.0