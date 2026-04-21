#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1wTrend_ATRStop_v1
Hypothesis: 6h Elder Ray Bull/Bear Power with 1-week EMA trend filter and ATR-based stop.
In bull markets (price > weekly EMA50): long when Bull Power > 0 and rising, short when Bear Power < 0 and falling.
In bear markets (price < weekly EMA50): only short retracements to EMA21 on 6h.
Uses ATR(22) stoploss (2.0x) and discrete position sizing (0.25) to limit fee drag and drawdown.
Designed to work in both bull and bear markets by adapting logic to weekly trend regime.
Timeframe: 6h, uses 1w HTF for trend filter.
Target: 50-150 total trades over 4 years = 12-37/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for EMA50 trend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # === 1w OHLC for EMA50 trend ===
    df_1w_close = df_1w['close'].values
    ema_50_1w = pd.Series(df_1w_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === 6h EMA21 for bear market retracement entries ===
    close = prices['close'].values
    ema_21_6h = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # === Elder Ray Power (13-period EMA) ===
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    high = prices['high'].values
    low = prices['low'].values
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = low - ema_13   # Bear Power = Low - EMA13
    
    # === ATR (22-period) for stoploss ===
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=22, min_periods=22).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(ema_21_6h[i]) 
            or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_trend_1w = ema_50_1w_aligned[i]
        ema_21 = ema_21_6h[i]
        bp = bull_power[i]
        bp_prev = bull_power[i-1] if i > 0 else bp
        br = bear_power[i]
        br_prev = bear_power[i-1] if i > 0 else br
        atr_val = atr[i]
        
        if position == 0:
            # Bull regime: price > weekly EMA50
            if price > ema_trend_1w:
                # Long when Bull Power > 0 and rising (strong buying pressure)
                if bp > 0 and bp > bp_prev:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
            # Bear regime: price < weekly EMA50
            else:
                # Short when Bear Power < 0 and falling (strong selling pressure) 
                # AND price is below 6h EMA21 (retracement entry in downtrend)
                if br < 0 and br < br_prev and price < ema_21:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
        
        elif position == 1:
            # Check stoploss (2.0x ATR)
            if price < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
            # Trend reversal exit (weekly trend turns bearish)
            elif price < ema_trend_1w:
                signals[i] = 0.0
                position = 0
            # Elder Ray exhaustion: Bull Power turns negative
            elif bp < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss (2.0x ATR)
            if price > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
            # Trend reversal exit (weekly trend turns bullish)
            elif price > ema_trend_1w:
                signals[i] = 0.0
                position = 0
            # Elder Ray exhaustion: Bear Power turns positive
            elif br > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_BullBearPower_1wTrend_ATRStop_v1"
timeframe = "6h"
leverage = 1.0