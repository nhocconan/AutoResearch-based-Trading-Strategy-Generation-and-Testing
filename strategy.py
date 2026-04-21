#!/usr/bin/env python3
"""
1d_Camarilla_R3_S3_Breakout_WeeklyTrend_Volume_v1
Hypothesis: Price breaking above Camarilla R3 or below S3 from prior 1d session on the daily timeframe captures strong institutional breakouts. Combined with weekly trend filter (price above/below 50-period EMA on 1w) and volume spike (>2.0x 20-period MA on 1d). Designed for low trade frequency (~10-25/year) to minimize fee drag and work in both bull (breakout continuation) and bear (breakdown continuation) regimes by requiring strong momentum confirmation and alignment with higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Camarilla levels, 1w for trend)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 2 or len(df_1w) < 50:
        return np.zeros(n)
    
    # === Camarilla levels from prior 1-day session (HLC of previous day) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R3, S3 levels (stronger breakout signals)
    camarilla_r3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_s3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # === Weekly trend filter: 50-period EMA on 1w ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === Volume spike filter (20-period on 1d) ===
    volume = df_1d['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(vol_ratio_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        vol_spike = vol_ratio_aligned[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        weekly_ema = ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above R3 + volume spike > 2.0 + price above weekly EMA (bullish trend)
            if price_close > r3 and vol_spike > 2.0 and price_close > weekly_ema:
                signals[i] = 0.25
                position = 1
                entry_price = price_close
            # Short: price breaks below S3 + volume spike > 2.0 + price below weekly EMA (bearish trend)
            elif price_close < s3 and vol_spike > 2.0 and price_close < weekly_ema:
                signals[i] = -0.25
                position = -1
                entry_price = price_close
        
        elif position != 0:
            # Stoploss: 3.0 * ATR(14) from entry (wider stop for less whipsaw on daily)
            # Calculate ATR using 1d data
            if i >= 14:
                high_1d = df_1d['high'].values
                low_1d = df_1d['low'].values
                close_1d = df_1d['close'].values
                
                tr1 = high_1d - low_1d
                tr2 = np.abs(high_1d - np.roll(close_1d, 1))
                tr3 = np.abs(low_1d - np.roll(close_1d, 1))
                tr = np.maximum(tr1, np.maximum(tr2, tr3))
                tr[0] = tr1[0]
                atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
                atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
                
                if position == 1:
                    if price_close < entry_price - 3.0 * atr_14_aligned[i]:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = 0.25
                else:  # position == -1
                    if price_close > entry_price + 3.0 * atr_14_aligned[i]:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = -0.25
            else:
                # Hold position until ATR is ready
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Camarilla_R3_S3_Breakout_WeeklyTrend_Volume_v1"
timeframe = "1d"
leverage = 1.0