#!/usr/bin/env python3
# 4H_RSI_Trend_Filter
# Hypothesis: Mean reversion within strong trends using RSI and trend filters.
# Long when: RSI < 30, price above 50-period EMA, and ADX > 25 (strong uptrend).
# Short when: RSI > 70, price below 50-period EMA, and ADX > 25 (strong downtrend).
# Uses 12h trend filter to avoid counter-trend trades in weak trends.
# Works in bull/bear by following trend and using RSI for mean reversion entries.
# Target: 20-30 trades/year per symbol.

name = "4H_RSI_Trend_Filter"
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
    
    # 4h indicators
    close_s = pd.Series(close)
    
    # EMA50 for trend direction
    ema50 = close_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    if len(gain) > 0:
        avg_gain[14] = np.mean(gain[:14])
        avg_loss[14] = np.mean(loss[:14])
        for i in range(15, len(close)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([np.full(14, np.nan), rsi])
    
    # ADX(14) for trend strength
    # +DM and -DM
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Wilder's smoothing
    atr = np.zeros_like(tr)
    if len(tr) > 0:
        atr[0] = tr[0]
        for i in range(1, len(tr)):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    # +DI and -DI
    plus_di = 100 * (np.convolve(plus_dm, np.ones(14)/14, mode='full')[:len(plus_dm)] / 
                     np.convolve(atr, np.ones(14)/14, mode='full')[:len(atr)])
    minus_di = 100 * (np.convolve(minus_dm, np.ones(14)/14, mode='full')[:len(minus_dm)] / 
                      np.convolve(atr, np.ones(14)/14, mode='full')[:len(atr)])
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = np.convolve(dx, np.ones(14)/14, mode='full')[:len(dx)]
    # Prepend NaN for alignment
    ema50 = np.concatenate([np.full(49, np.nan), ema50])
    rsi = np.concatenate([np.full(14, np.nan), rsi])
    adx = np.concatenate([np.full(13, np.nan), adx])
    
    # 12h trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_12h = close_12h > ema50_12h
    downtrend_12h = close_12h < ema50_12h
    
    # Align 12h trend to 4h
    uptrend_12h_aligned = align_htf_to_ltf(prices, df_12h, uptrend_12h.astype(float))
    downtrend_12h_aligned = align_htf_to_ltf(prices, df_12h, downtrend_12h.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50[i]) or np.isnan(rsi[i]) or np.isnan(adx[i]) or
            np.isnan(uptrend_12h_aligned[i]) or np.isnan(downtrend_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_above_ema = close[i] > ema50[i]
        price_below_ema = close[i] < ema50[i]
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        strong_trend = adx[i] > 25
        
        uptrend_12h = uptrend_12h_aligned[i] > 0.5
        downtrend_12h = downtrend_12h_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: 12h uptrend + strong 4h trend + RSI oversold + price above EMA50
            if uptrend_12h and strong_trend and rsi_oversold and price_above_ema:
                signals[i] = 0.25
                position = 1
            # Enter short: 12h downtrend + strong 4h trend + RSI overbought + price below EMA50
            elif downtrend_12h and strong_trend and rsi_overbought and price_below_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit conditions: trend weakens or RSI normalizes
            if not uptrend_12h or not strong_trend or rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions: trend weakens or RSI normalizes
            if not downtrend_12h or not strong_trend or rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals