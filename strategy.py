#!/usr/bin/env python3
"""
6h_TRIX_Volume_Regime
Hypothesis: TRIX (triple smoothed ROC) identifies momentum shifts with less whipsaw, combined with volume confirmation and regime filter (ADX > 25) to capture trends in both bull and bear markets. Designed for low trade frequency (15-30/year) on 6h timeframe.
"""

name = "6h_TRIX_Volume_Regime"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate TRIX (15,9,9) - triple smoothed 1-period ROC
    # ROC(1) = (close/tclose[-1] - 1) * 100
    roc = np.diff(np.log(close), prepend=np.log(close[0])) * 100
    # Threefold EMA smoothing
    ema1 = pd.Series(roc).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema3 = pd.Series(ema2).ewm(span=9, adjust=False, min_periods=9).mean().values
    trix = ema3
    
    # Calculate ADX(14) for regime filter
    # +DM, -DM, TR
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    # Pad first element
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    tr = np.concatenate([[0], tr])
    
    # Smoothed averages
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    # Get 12h trend filter (EMA 50)
    df_12h = get_htf_data(prices, '12h')
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if position == 0:
            # LONG: TRIX turns positive with volume confirmation and ADX > 25 (trending market)
            if trix[i] > 0 and trix[i-1] <= 0 and volume_confirm[i] and adx[i] > 25:
                # Additional filter: only take long if price above 12h EMA50 (uptrend filter)
                if close[i] > ema_50_12h_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            # SHORT: TRIX turns negative with volume confirmation and ADX > 25 (trending market)
            elif trix[i] < 0 and trix[i-1] >= 0 and volume_confirm[i] and adx[i] > 25:
                # Additional filter: only take short if price below 12h EMA50 (downtrend filter)
                if close[i] < ema_50_12h_aligned[i]:
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX turns negative or ADX < 20 (range market)
            if trix[i] < 0 or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX turns positive or ADX < 20 (range market)
            if trix[i] > 0 or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals