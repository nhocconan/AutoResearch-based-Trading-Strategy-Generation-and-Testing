#!/usr/bin/env python3
# 1H_4H_1D_Trend_Momentum_With_Reversal_Filter
# Hypothesis: In 1h, trade momentum in direction of 4h trend only when 1d trend confirms and momentum is not exhausted.
# Uses 4h EMA20/50 crossover for trend, 1d EMA50 for higher timeframe bias, and 1h RSI(2) for short-term mean reversion entry.
# Long when: 4h EMA20 > EMA50, 1d close > EMA50, and RSI(2) < 10 (oversold pullback in uptrend).
# Short when: 4h EMA20 < EMA50, 1d close < EMA50, and RSI(2) > 90 (overbought rally in downtrend).
# Avoids chop via ADX(14) < 20 filter on 4h.
# Target: 15-30 trades/year per symbol by requiring multi-timeframe alignment and extreme RSI.

name = "1H_4H_1D_Trend_Momentum_With_Reversal_Filter"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1h RSI(2) for short-term mean reversion entry
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    avg_loss = loss.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # 4h EMA20 and EMA50 for trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Trend: EMA20 > EMA50 = uptrend, EMA20 < EMA50 = downtrend
    ema20_50_cross = ema20_4h - ema50_4h
    ema20_gt_ema50 = ema20_50_cross > 0
    ema20_lt_ema50 = ema20_50_cross < 0
    # Align 4h trend to 1h
    ema20_gt_ema50_aligned = align_htf_to_ltf(prices, df_4h, ema20_gt_ema50.astype(float))
    ema20_lt_ema50_aligned = align_htf_to_ltf(prices, df_4h, ema20_lt_ema50.astype(float))
    
    # 4h ADX(14) for trend strength (avoid chop)
    # Calculate +DM, -DM, TR
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Wilder's smoothing for ATR
    atr = np.zeros_like(tr)
    if len(tr) > 0:
        atr[0] = tr[0]
        for i in range(1, len(tr)):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    # +DI and -DI
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean() / 
                     pd.Series(atr).ewm(alpha=1/14, adjust=False).mean())
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean() / 
                      pd.Series(atr).ewm(alpha=1/14, adjust=False).mean())
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    # Prepend NaN for alignment (lost first bar in calculations)
    adx = np.concatenate([np.full(1, np.nan), adx])
    # Align 4h ADX to 1h
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx)
    
    # 1d EMA50 for higher timeframe trend bias
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    daily_uptrend = close_1d > ema50_1d
    daily_downtrend = close_1d < ema50_1d
    # Align 1d trend to 1h
    daily_uptrend_aligned = align_htf_to_ltf(prices, df_1d, daily_uptrend.astype(float))
    daily_downtrend_aligned = align_htf_to_ltf(prices, df_1d, daily_downtrend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data for all indicators
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(rsi_values[i]) or np.isnan(ema20_gt_ema50_aligned[i]) or 
            np.isnan(ema20_lt_ema50_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(daily_uptrend_aligned[i]) or np.isnan(daily_downtrend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # 4h trend strength filter: avoid choppy markets
        strong_trend = adx_aligned[i] > 20
        
        rsi_val = rsi_values[i]
        ema20_gt = ema20_gt_ema50_aligned[i] > 0.5
        ema20_lt = ema20_lt_ema50_aligned[i] > 0.5
        daily_up = daily_uptrend_aligned[i] > 0.5
        daily_down = daily_downtrend_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: 4h uptrend + 1d uptrend + RSI oversold + strong trend
            if ema20_gt and daily_up and rsi_val < 10 and strong_trend:
                signals[i] = 0.20
                position = 1
            # Enter short: 4h downtrend + 1d downtrend + RSI overbought + strong trend
            elif ema20_lt and daily_down and rsi_val > 90 and strong_trend:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: trend breaks or RSI normalizes
            if not ema20_gt or not daily_up or rsi_val > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: trend breaks or RSI normalizes
            if not ema20_lt or not daily_down or rsi_val < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals