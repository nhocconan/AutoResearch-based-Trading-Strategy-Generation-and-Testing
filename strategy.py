#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_ChopFilter_v1
Hypothesis: On 1d timeframe, KAMA trend direction combined with RSI extremes and choppiness regime filter produces low-frequency, high-probability trades. KAMA adapts to market noise, RSI identifies overextended conditions for mean reversion within trend, and choppiness filter avoids whipsaws in strong trends. Target: 30-80 total trades over 4 years (7-20/year) with discrete sizing to minimize fee drag.
"""

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
    
    # Load 1w data ONCE before loop for HTF trend context
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA50 for higher timeframe trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 1d KAMA (adaptive moving average) - trend direction
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)  # will adjust below
    # Recalculate volatility properly: sum of absolute changes over 10 periods
    volatility = np.zeros_like(close)
    for i in range(10, len(close)):
        volatility[i] = np.sum(np.abs(np.diff(close[i-10:i+1])))
    # Avoid division by zero
    er = np.where(volatility > 0, change / volatility, 0)
    # Smoothing constants
    fastest = 2 / (2 + 1)   # EMA(2)
    slowest = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fastest - slowest) + slowest) ** 2
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    kama_aligned = kama  # already 1d
    
    # 1d RSI(14) for mean reversion signals
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Pad beginning with NaN
    rsi = np.concatenate([[np.nan] * 14, rsi])
    
    # 1d Choppiness Index(14) - regime filter
    # True Range
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    # ATR(14)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Choppiness Index
    chop = np.where(
        (atr > 0) & (hh > ll),
        100 * np.log10(np.sum(tr) / (hh - ll)) / np.log10(14),
        50  # default when undefined
    )
    # For proper calculation, compute per period
    chop = np.full_like(close, np.nan)
    for i in range(14, len(close)):
        tr_sum = np.sum(tr[i-13:i+1])
        if hh[i] > ll[i] and atr[i] > 0:
            chop[i] = 100 * np.log10(tr_sum / (hh[i] - ll[i])) / np.log10(14)
        else:
            chop[i] = 50.0
    
    # 1d volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for 1w EMA, 30 for KAMA SC, 14 for RSI/chop, 20 for volume)
    start_idx = max(50, 30, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(kama[i]) or
            np.isnan(rsi[i]) or
            np.isnan(chop[i]) or
            np.isnan(vol_ma_20[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 1w trend filter (EMA50) - only trade in direction of higher timeframe trend
        uptrend_1w = close[i] > ema_50_1w_aligned[i]
        downtrend_1w = close[i] < ema_50_1w_aligned[i]
        
        # KAMA direction - short term trend
        kama_up = close[i] > kama[i]
        kama_down = close[i] < kama[i]
        
        # RSI extremes for mean reversion
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        # Choppiness regime: chop > 61.8 = ranging (good for mean reversion)
        chop_ranging = chop[i] > 61.8
        
        # Volume confirmation
        volume_confirm = volume[i] > 1.5 * vol_ma_20[i]
        
        # Long logic: KAMA uptrend + RSI oversold + ranging market + volume
        if kama_up and rsi_oversold and chop_ranging and volume_confirm:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short logic: KAMA downtrend + RSI overbought + ranging market + volume
        elif kama_down and rsi_overbought and chop_ranging and volume_confirm:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: opposite RSI extreme or loss of KAMA alignment
        elif position == 1 and (rsi[i] > 70 or not kama_up):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (rsi[i] < 30 or not kama_down):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Direction_RSI_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0