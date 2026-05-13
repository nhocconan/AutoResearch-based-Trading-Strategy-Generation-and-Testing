#!/usr/bin/env python3
"""
4h_Keltner_Channel_Breakout_Trend_Volume_v2
Hypothesis: Keltner Channel (20, ATR=2) breakouts with 4h trend (EMA50) and volume confirmation work in both bull and bear markets.
Breakout above upper band with uptrend and volume spike = long.
Breakdown below lower band with downtrend and volume spike = short.
Exit on opposite band touch or trend reversal. Uses 1d trend filter for higher timeframe bias.
Target: 20-50 trades/year per symbol.
"""

name = "4h_Keltner_Channel_Breakout_Trend_Volume_v2"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Keltner Channel: 20 EMA, ATR(20) * 2
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[0.0], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_20 = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    upper_band = ema_20 + 2 * atr_20
    lower_band = ema_20 - 2 * atr_20
    
    # 4h trend: EMA50
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_4h = close > ema_50
    downtrend_4h = close < ema_50
    
    # 1d trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1d = df_1d['close'].values > ema_50_1d
    downtrend_1d = df_1d['close'].values < ema_50_1d
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 2.0 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        upband = upper_band[i]
        lowband = lower_band[i]
        uptrend = uptrend_4h[i]
        downtrend = downtrend_4h[i]
        uptrend_htf = uptrend_1d_aligned[i]
        downtrend_htf = downtrend_1d_aligned[i]
        vol_conf = volume_conf[i]
        
        if position == 0:
            # LONG: break above upper band, 4h uptrend, 1d uptrend filter, volume confirmation
            if close[i] > upband and uptrend and uptrend_htf and vol_conf:
                signals[i] = 0.25
                position = 1
            # SHORT: break below lower band, 4h downtrend, 1d downtrend filter, volume confirmation
            elif close[i] < lowband and downtrend and downtrend_htf and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: touch lower band or 4h trend turns down
            if close[i] < lowband or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: touch upper band or 4h trend turns up
            if close[i] > upband or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals