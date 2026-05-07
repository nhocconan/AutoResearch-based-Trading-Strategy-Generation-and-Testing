#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend with weekly trend filter and volume confirmation.
# Long when KAMA indicates uptrend AND price > weekly EMA34 AND volume > 1.5x average.
# Short when KAMA indicates downtrend AND price < weekly EMA34 AND volume > 1.5x average.
# KAMA adapts to market noise, reducing whipsaw in ranging markets.
# Weekly EMA34 filter ensures alignment with higher timeframe trend.
# Volume confirmation adds momentum validation. Designed for 10-20 trades/year to minimize fee drag.
# Works in both bull and bear markets by following adaptive trend and weekly trend direction.
name = "1d_KAMA_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Weekly trend filter: 34-period EMA on close
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # KAMA (Kaufman Adaptive Moving Average) - 10-period ER, 2 and 30 SC
    # ER = |net change| / sum(|changes|)
    change = np.abs(np.diff(close, prepend=close[0]))
    dir = np.abs(np.diff(close, k=10, prepend=close[:10]))
    vol = np.sum(np.lib.stride_tricks.sliding_window_view(change, 10), axis=1)
    # Pad vol to match length
    vol = np.concatenate([np.full(9, np.nan), vol])
    er = np.where(vol > 0, dir / vol, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # Initialize KAMA
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # Start after first 10 periods
    for i in range(10, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # 1d volume average for spike detection
    vol_ma = np.convolve(volume, np.ones(20)/20, mode='same')
    vol_ma[:10] = np.nan
    vol_ma[-10:] = np.nan
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Sufficient warmup for KAMA and EMA calculation
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(kama[i]) or 
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # KAMA trend: price above/below KAMA
        kama_uptrend = close[i] > kama[i]
        kama_downtrend = close[i] < kama[i]
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema_34_1w_aligned[i]
        weekly_downtrend = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long condition: KAMA uptrend, weekly uptrend, volume spike
            long_condition = kama_uptrend and weekly_uptrend and vol_spike[i]
            # Short condition: KAMA downtrend, weekly downtrend, volume spike
            short_condition = kama_downtrend and weekly_downtrend and vol_spike[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: KAMA downtrend or weekly downtrend
            if (not kama_uptrend) or (not weekly_uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: KAMA uptrend or weekly uptrend
            if (not kama_downtrend) or (not weekly_downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals