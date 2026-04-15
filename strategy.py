#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA trend direction + RSI(14) mean reversion + volume spike filter
# Long when KAMA bullish (price > KAMA) + RSI < 30 (oversold) + volume > 1.5x 20-period avg
# Short when KAMA bearish (price < KAMA) + RSI > 70 (overbought) + volume > 1.5x 20-period avg
# Uses discrete position sizing (0.25) to control drawdown and minimize fee drag.
# KAMA adapts to market noise, reducing whipsaws in both bull and bear markets.
# Volume threshold (1.5x) targets ~15-35 trades/year on 12h timeframe to avoid overtrading.
# RSI extremes provide mean-reversion edge in ranging markets, KAMA filters trend.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # === 1d Indicator: EMA34 for trend filter ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === KAMA (Kaufman Adaptive Moving Average) ===
    # ER = |net change| / sum(|changes|)
    # Smoothest ER constant = 2/(fast+1) - 2/(slow+1)
    fast_sc = 2 / (2 + 1)      # EMA(2)
    slow_sc = 2 / (30 + 1)     # EMA(30)
    sc = fast_sc - slow_sc
    
    # Calculate ER over 10 periods
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close, prepend=close[0]))
    er = np.zeros_like(change)
    for i in range(10, len(change)):
        net_change = abs(change[i] - change[i-10])
        total_change = np.sum(volatility[i-9:i+1])
        er[i] = net_change / total_change if total_change != 0 else 0
    
    # Smoothing constant
    sc = (er * sc + slow_sc) ** 2
    # KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === RSI(14) ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(34, 20, 14) + 5  # EMA34 + KAMA(10) + RSI(14) + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(kama[i]) or
            np.isnan(rsi[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # === LONG CONDITIONS ===
        # 1. Price above KAMA (bullish trend)
        # 2. RSI < 30 (oversold)
        # 3. Volume confirmation
        if (close[i] > kama[i]) and \
           (rsi[i] < 30) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price below KAMA (bearish trend)
        # 2. RSI > 70 (overbought)
        # 3. Volume confirmation
        elif (close[i] < kama[i]) and \
             (rsi[i] > 70) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_KAMA_RSI_Volume_Filter_v1"
timeframe = "12h"
leverage = 1.0