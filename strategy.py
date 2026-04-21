#!/usr/bin/env python3
"""
4h_1d_BollingerBreakout_TrendFilter
Hypothesis: Bollinger Band breakouts with volume confirmation and trend filter (1d EMA34) work in both bull and bear markets by capturing strong momentum moves. The trend filter ensures we only trade in the direction of the daily trend, reducing false signals. Designed for low trade frequency (target: 20-50/year) to minimize fee drag in 4h timeframe. Uses Bollinger Bands (20,2) for volatility-based breakouts, volume surge for confirmation, and daily EMA34 for trend direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data once for EMA34 trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    close_daily = df_daily['close'].values
    ema34_daily = np.zeros_like(close_daily)
    ema34_daily[0] = close_daily[0]
    alpha = 2.0 / (34 + 1)
    for i in range(1, len(close_daily)):
        ema34_daily[i] = alpha * close_daily[i] + (1 - alpha) * ema34_daily[i-1]
    
    # Align daily EMA34 to 4h timeframe
    ema34_daily_aligned = align_htf_to_ltf(prices, df_daily, ema34_daily)
    
    # Main timeframe data (4h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2) on 4h
    bb_period = 20
    bb_std = 2.0
    sma = np.zeros_like(close)
    bb_up = np.zeros_like(close)
    bb_dn = np.zeros_like(close)
    
    for i in range(n):
        if i < bb_period:
            sma[i] = np.mean(close[:i+1])
        else:
            sma[i] = np.mean(close[i-bb_period+1:i+1])
        
        if i < bb_period:
            bb_std_dev = np.std(close[:i+1])
        else:
            bb_std_dev = np.std(close[i-bb_period+1:i+1])
        
        bb_up[i] = sma[i] + bb_std * bb_std_dev
        bb_dn[i] = sma[i] - bb_std * bb_std_dev
    
    # Volume filter: current volume > 2.0x 20-period average
    volume_avg = np.zeros_like(volume)
    for i in range(len(volume)):
        if i >= 20:
            volume_avg[i] = np.mean(volume[i-20:i])
        else:
            volume_avg[i] = np.mean(volume[:i+1]) if i > 0 else volume[i]
    volume_filter = volume > (2.0 * volume_avg)
    
    # ATR for stoploss (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr = np.zeros_like(close)
    for i in range(len(tr)):
        if i < 14:
            atr[i] = np.mean(tr[:i+1])
        else:
            atr[i] = np.mean(tr[i-14:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(34, n):  # Start after EMA warmup
        # Skip if NaN in critical values
        if np.isnan(ema34_daily_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema34 = ema34_daily_aligned[i]
        bb_upper = bb_up[i]
        bb_lower = bb_dn[i]
        vol_ok = volume_filter[i]
        atr_val = atr[i]
        
        # Stoploss: 2.5 * ATR from entry
        if position == 1 and price < entry_price - 2.5 * atr_val:
            signals[i] = 0.0
            position = 0
            continue
        elif position == -1 and price > entry_price + 2.5 * atr_val:
            signals[i] = 0.0
            position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper Bollinger Band with volume and uptrend (price > daily EMA34)
            if price > bb_upper and vol_ok and price > ema34:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below lower Bollinger Band with volume and downtrend (price < daily EMA34)
            elif price < bb_lower and vol_ok and price < ema34:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price returns to middle Bollinger Band (mean reversion) or breaks below lower band (failed breakout)
            if price < sma[i] or price < bb_lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to middle Bollinger Band (mean reversion) or breaks above upper band (failed breakdown)
            if price > sma[i] or price > bb_upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_BollingerBreakout_TrendFilter"
timeframe = "4h"
leverage = 1.0