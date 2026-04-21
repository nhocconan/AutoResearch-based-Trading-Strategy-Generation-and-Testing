#!/usr/bin/env python3
"""
4h_KAMA_Direction_VolumeSpike_ATRStop_v1
Hypothesis: On 4h timeframe, Kaufman Adaptive Moving Average (KAMA) trend direction combined with volume spike confirmation and ATR-based stoploss captures sustained momentum moves while minimizing whipsaws. Uses 1-week EMA50 as higher timeframe trend filter. Designed for low trade frequency (target: 20-50/year) to reduce fee drag and improve generalization across bull/bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for higher timeframe trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1-week EMA50 for higher timeframe trend filter ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === Kaufman Adaptive Moving Average (KAMA) on 4h timeframe ===
    close = prices['close'].values
    direction = np.abs(np.diff(close, 10))  # net change over 10 periods
    volatility = np.sum(np.abs(np.diff(close, 1)), axis=0) if len(close) > 1 else np.zeros_like(close)
    # Fix array shapes for rolling sum
    volatility_full = np.zeros(n)
    for i in range(1, n):
        volatility_full[i] = volatility_full[i-1] + np.abs(close[i] - close[i-1])
        if i >= 1:
            volatility_full[i] -= np.abs(close[i-10] - close[i-11]) if i >= 11 else 0
    volatility_full[:10] = np.sum(np.abs(np.diff(close[:11], 1))) if len(close) >= 11 else 0
    er = np.where(volatility_full != 0, direction / volatility_full, 0)
    sc = (er * (0.6 - 0.06) + 0.06) ** 2  # smoothing constant
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    kama_series = pd.Series(kama)
    kama_values = kama_series.ewm(span=2, adjust=False).mean().values  # smooth KAMA
    
    # === ATR for volatility filtering and stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # === Volume confirmation: 4h volume > 2.0 * 20-period MA ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(kama_values[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        ema_50_1w = ema_50_1w_aligned[i]
        kama_val = kama_values[i]
        atr_val = atr[i]
        vol_conf = volume_confirmed[i]
        
        if position == 0:
            # Long: price above KAMA + above weekly EMA50 + volume confirmation
            if price_close > kama_val and price_close > ema_50_1w and vol_conf:
                signals[i] = 0.25
                position = 1
                entry_price = price_close
            # Short: price below KAMA + below weekly EMA50 + volume confirmation
            elif price_close < kama_val and price_close < ema_50_1w and vol_conf:
                signals[i] = -0.25
                position = -1
                entry_price = price_close
        
        elif position != 0:
            # ATR-based stoploss and trend exit
            if position == 1:
                # Stoploss: 2.5 * ATR below entry
                stop_price = entry_price - 2.5 * atr_val
                # Exit if price hits stop or trend weakens (price below KAMA)
                if price_low < stop_price or price_close < kama_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Stoploss: 2.5 * ATR above entry
                stop_price = entry_price + 2.5 * atr_val
                # Exit if price hits stop or trend weakens (price above KAMA)
                if price_high > stop_price or price_close > kama_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_KAMA_Direction_VolumeSpike_ATRStop_v1"
timeframe = "4h"
leverage = 1.0