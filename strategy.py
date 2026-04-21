#!/usr/bin/env python3
"""
1d_Camarilla_R3_S3_Breakout_1wTrend_v1
Hypothesis: On 1d timeframe, price breaking above Camarilla R3 or below S3 levels from prior 1w session captures multi-day institutional breakouts. Combined with 1w EMA50 trend filter and ATR-based stoploss. Designed for very low trade frequency (<25/year) to minimize fee drag and work in both bull (breakout continuation) and bear (breakdown continuation) regimes by following the weekly trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for Camarilla levels and EMA trend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1-week EMA50 for trend filter ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === Camarilla levels from prior 1-week session (HLC of previous week) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Camarilla R3, S3, R4, S4
    camarilla_r3 = close_1w + (high_1w - low_1w) * 1.1 / 4
    camarilla_s3 = close_1w - (high_1w - low_1w) * 1.1 / 4
    camarilla_r4 = close_1w + (high_1w - low_1w) * 1.1 / 2
    camarilla_s4 = close_1w - (high_1w - low_1w) * 1.1 / 2
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4)
    
    # === ATR for volatility and stoploss (14-period on 1d) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        ema_50 = ema_50_1w_aligned[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        r4 = camarilla_r4_aligned[i]
        s4 = camarilla_s4_aligned[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: price breaks above R3 (bullish breakout) + above 1w EMA50 (uptrend)
            if price_close > r3 and price_close > ema_50:
                signals[i] = 0.25
                position = 1
                entry_price = price_close
            # Short: price breaks below S3 (bearish breakdown) + below 1w EMA50 (downtrend)
            elif price_close < s3 and price_close < ema_50:
                signals[i] = -0.25
                position = -1
                entry_price = price_close
        
        elif position != 0:
            # Stoploss: 2 * ATR from entry
            if position == 1:
                if price_close < entry_price - 2.0 * atr_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price_close > entry_price + 2.0 * atr_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_R3_S3_Breakout_1wTrend_v1"
timeframe = "1d"
leverage = 1.0