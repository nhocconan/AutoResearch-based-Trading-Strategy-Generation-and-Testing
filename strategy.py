#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_TrendVolume_ATRStop_v1
Hypothesis: Price breaking above/below 20-period Donchian channel on 4h captures momentum with institutional follow-through. Combined with 4h EMA50 trend filter and volume spike (>1.8x 20-period MA) to avoid false breakouts. Uses ATR-based trailing stop (2.5x ATR) for risk control. Designed for low trade frequency (20-40/year) to minimize fee drag and work in both bull (breakout continuation) and bear (breakdown continuation) regimes by requiring trend alignment and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for higher timeframe trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === Donchian Channel (20-period) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Upper band: highest high over 20 periods
    upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low over 20 periods
    lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === EMA50 trend filter (4h) ===
    close_series = pd.Series(close)
    ema50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === Volume spike filter (20-period) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    # === ATR (14-period) for stoploss ===
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
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(ema50[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        vol_spike = vol_ratio[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian + price > EMA50 (uptrend) + volume spike > 1.8
            if price_close > upper[i] and price_close > ema50[i] and vol_spike > 1.8:
                signals[i] = 0.25
                position = 1
                entry_price = price_close
            # Short: price breaks below lower Donchian + price < EMA50 (downtrend) + volume spike > 1.8
            elif price_close < lower[i] and price_close < ema50[i] and vol_spike > 1.8:
                signals[i] = -0.25
                position = -1
                entry_price = price_close
        
        elif position != 0:
            # Trailing stoploss: 2.5 * ATR from entry
            if position == 1:
                if price_close < entry_price - 2.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price_close > entry_price + 2.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_TrendVolume_ATRStop_v1"
timeframe = "4h"
leverage = 1.0