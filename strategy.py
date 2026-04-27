#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA + RSI + Choppiness regime filter
# KAMA adapts to market noise, avoiding whipsaws in sideways markets.
# RSI(14) provides overbought/oversold signals filtered by KAMA direction.
# Choppiness index (CHOP) identifies ranging (CHOP > 61.8) vs trending (CHOP < 38.2) markets.
# In ranging markets, we mean-revert at RSI extremes; in trending markets, we follow KAMA direction.
# Weekly trend filter ensures we only take trades aligned with higher timeframe momentum.
# Target: 20-60 trades per year to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly EMA(40) for trend filter
    ema_40_1w = pd.Series(df_1w['close'].values).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema_40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_40_1w)
    
    # Calculate daily KAMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Efficiency Ratio for KAMA
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d))
    er = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i == 0:
            er[i] = 0
        else:
            dir = np.abs(close_1d[i] - close_1d[i-10]) if i >= 10 else np.abs(close_1d[i] - close_1d[0])
            vol = np.sum(volatility[max(0, i-9):i+1]) if i >= 10 else np.sum(volatility[1:i+1])
            er[i] = dir / vol if vol != 0 else 0
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    
    # KAMA calculation
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Calculate daily RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close_1d)
    avg_loss = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i < 14:
            if i == 0:
                avg_gain[i] = gain[i]
                avg_loss[i] = loss[i]
            else:
                avg_gain[i] = (avg_gain[i-1] * (i-1) + gain[i]) / i if i > 0 else gain[i]
                avg_loss[i] = (avg_loss[i-1] * (i-1) + loss[i]) / i if i > 0 else loss[i]
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Calculate daily Choppiness Index
    atr = np.zeros_like(close_1d)
    tr1 = np.abs(np.subtract(high[1:], low[1:]))
    tr2 = np.abs(np.subtract(high[1:], close[:-1]))
    tr3 = np.abs(np.subtract(low[1:], close[:-1]))
    tr = np.concatenate([[np.max([high[0], low[0], close[0]]) - np.min([high[0], low[0], close[0]])], 
                         np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i < 14:
            atr[i] = np.mean(tr[max(0, i-13):i+1]) if i > 0 else tr[0]
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Calculate highest high and lowest low over 14 periods
    hh = np.zeros_like(close_1d)
    ll = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        start_idx = max(0, i-13)
        hh[i] = np.max(high[start_idx:i+1])
        ll[i] = np.min(low[start_idx:i+1])
    
    # Chop calculation
    chop = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i < 13:
            chop[i] = 50  # neutral
        else:
            sum_tr = np.sum(tr[max(0, i-13):i+1])
            max_range = hh[i] - ll[i]
            chop[i] = 100 * np.log10(sum_tr / max_range) / np.log10(14) if max_range != 0 else 50
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position
    
    # Warmup: need weekly EMA (40), daily KAMA/RSI/CHOP (14)
    start_idx = max(40, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_40_1w_aligned[i]) or np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        chop_val = chop_aligned[i]
        weekly_ema = ema_40_1w_aligned[i]
        
        # Determine market regime
        is_ranging = chop_val > 61.8
        is_trending = chop_val < 38.2
        
        # Weekly trend filter
        weekly_uptrend = price > weekly_ema
        weekly_downtrend = price < weekly_ema
        
        if position == 0:
            # In ranging markets: mean reversion at RSI extremes
            if is_ranging:
                if rsi_val < 30 and weekly_uptrend:  # oversold in up trend
                    signals[i] = size
                    position = 1
                elif rsi_val > 70 and weekly_downtrend:  # overbought in down trend
                    signals[i] = -size
                    position = -1
            # In trending markets: follow KAMA direction with weekly filter
            elif is_trending:
                if price > kama_val and weekly_uptrend:
                    signals[i] = size
                    position = 1
                elif price < kama_val and weekly_downtrend:
                    signals[i] = -size
                    position = -1
        elif position == 1:
            # Exit long: RSI overbought OR price crosses below KAMA OR weekly trend turns down
            if rsi_val > 70 or price < kama_val or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: RSI oversold OR price crosses above KAMA OR weekly trend turns up
            if rsi_val < 30 or price > kama_val or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_KAMA_RSI_Chop_Regime"
timeframe = "1d"
leverage = 1.0