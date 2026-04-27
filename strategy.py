#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_ChopFilter_v1
Hypothesis: KAMA adapts to market efficiency, filtering noise. Combined with RSI extremes and Choppiness Index regime filter (CHOP > 61.8 = range), this captures trending moves while avoiding whipsaws. Weekly trend filter ensures alignment with higher timeframe momentum. Works in both bull (catch trends) and bear (avoid false reversals in ranging markets).
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
    
    # Calculate KAMA (adaptive moving average)
    # Efficiency Ratio: ER = |close - close[10]| / sum(|close - close[-1]|) over 10 periods
    change = np.abs(close - np.roll(close, 10))
    change[0:10] = 0  # first 10 bars
    volatility = np.abs(np.diff(close, prepend=close[0]))
    volatility_sum = pd.Series(volatility).rolling(window=10, min_periods=10).sum().values
    er = np.where(volatility_sum > 0, change / volatility_sum, 0)
    # Smoothing constants: sc = [ER * (2/(2+1) - 2/(30+1)) + 2/(30+1)]^2
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14)
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr1 = np.maximum(tr1, np.absolute(low - np.roll(close, 1)))
    tr1[0] = high[0] - low[0]
    tr_sum = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
    atr = pd.Series(high - low).rolling(window=14, min_periods=14).mean().values
    chop = 100 * np.log10(tr_sum / (atr * 14)) / np.log10(14)
    
    # Get weekly trend filter: price vs 1w EMA50
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    # Align indicators
    kama_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), kama)
    rsi_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), rsi)
    chop_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), chop)
    volume_confirm_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), volume_confirm)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    start_idx = max(10, 14, 20, 50)  # KAMA(10), CHOP(14), VOL(20), EMA50(50)
    
    for i in range(start_idx, n):
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(volume_confirm_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i])):
            continue
        
        close_val = close[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        chop_val = chop_aligned[i]
        vol_conf = volume_confirm_aligned[i]
        ema50 = ema50_1w_aligned[i]
        
        if position == 0:
            # Long: price > KAMA, RSI > 50 (bullish momentum), not choppy, volume confirms, above weekly EMA50
            if close_val > kama_val and rsi_val > 50 and chop_val < 61.8 and vol_conf and close_val > ema50:
                signals[i] = size
                position = 1
            # Short: price < KAMA, RSI < 50 (bearish momentum), not choppy, volume confirms, below weekly EMA50
            elif close_val < kama_val and rsi_val < 50 and chop_val < 61.8 and vol_conf and close_val < ema50:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit: price crosses below KAMA OR RSI < 40
            if close_val < kama_val or rsi_val < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit: price crosses above KAMA OR RSI > 60
            if close_val > kama_val or rsi_val > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_KAMA_Trend_RSI_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0