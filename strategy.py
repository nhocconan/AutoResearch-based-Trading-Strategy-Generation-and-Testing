#!/usr/bin/env python3
"""
1d_1w_kama_rsi_chop_v1
Strategy: Daily KAMA trend with RSI momentum and Choppiness regime filter
Timeframe: 1d
Leverage: 1.0
Hypothesis: Uses Kaufman Adaptive Moving Average (KAMA) for trend direction on daily timeframe,
filtered by RSI momentum and Choppiness Index regime to avoid whipsaws. KAMA adapts to market
efficiency, reducing lag in trends while staying flat in ranges. Combined with RSI(14) for
momentum confirmation and Choppiness Index > 61.8 to identify ranging markets where we avoid
trend trades. Designed to work in both bull (trend following) and bear (avoid false signals)
markets by focusing on high-efficiency trending periods only.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data for higher timeframe filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === KAMA Calculation (Adaptive Moving Average) ===
    # Efficiency Ratio: |net change| / sum(|abs changes|) over ER period
    er_period = 10
    fast_sc = 2 / (2 + 1)   # SC for fastest EMA
    slow_sc = 2 / (30 + 1)  # SC for slowest EMA
    
    change = np.abs(np.diff(close, prepend=close[0]))
    abs_change = np.abs(np.diff(close, prepend=close[0]))
    
    # Vectorized ER calculation
    er = np.zeros(n)
    for i in range(er_period, n):
        net_change = abs(close[i] - close[i-er_period])
        total_change = np.sum(change[i-er_period+1:i+1])
        if total_change > 0:
            er[i] = net_change / total_change
        else:
            er[i] = 0
    
    # Smoothing Constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === RSI Calculation ===
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[rsi_period] = np.mean(gain[1:rsi_period+1])
    avg_loss[rsi_period] = np.mean(loss[1:rsi_period+1])
    
    for i in range(rsi_period+1, n):
        avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
        avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # === Choppiness Index Calculation ===
    chop_period = 14
    atr = np.zeros(n)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[high[0] - low[0]], tr])
    
    # True Range sum over period
    tr_sum = np.zeros(n)
    for i in range(chop_period, n):
        tr_sum[i] = np.sum(tr[i-chop_period+1:i+1])
    
    # Highest high and lowest low over period
    hh = np.zeros(n)
    ll = np.zeros(n)
    for i in range(chop_period-1, n):
        hh[i] = np.max(high[i-chop_period+1:i+1])
        ll[i] = np.min(low[i-chop_period+1:i+1])
    
    # Chop calculation: 100 * log10(TRsum / (HH - LL)) / log10(period)
    hh_ll = hh - ll
    # Avoid division by zero
    hh_ll_safe = np.where(hh_ll == 0, 1e-10, hh_ll)
    chop = 100 * np.log10(tr_sum / hh_ll_safe) / np.log10(chop_period)
    chop = np.where(tr_sum > 0, chop, 50)  # Default to middle when no TR
    
    # === Weekly Trend Filter (using EMA on weekly data) ===
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # === Signal Generation ===
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after all indicators are warm
    start_idx = max(100, er_period, rsi_period+1, chop_period)
    
    for i in range(start_idx, n):
        # Skip if any data invalid
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend direction from KAMA
        kama_uptrend = price_close > kama[i]
        kama_downtrend = price_close < kama[i]
        
        # Momentum confirmation from RSI
        rsi_overbought = rsi[i] > 70
        rsi_oversold = rsi[i] < 30
        rsi_momentum_up = rsi[i] > 50  # Bullish momentum
        rsi_momentum_down = rsi[i] < 50  # Bearish momentum
        
        # Regime filter: Avoid trending markets when too choppy
        # Chop > 61.8 = ranging market (avoid trend trades)
        # Chop < 38.2 = trending market (favor trend trades)
        chop_ranging = chop[i] > 61.8
        chop_trending = chop[i] < 38.2
        
        # Weekly trend alignment
        weekly_uptrend = price_close > ema_20_1w_aligned[i]
        weekly_downtrend = price_close < ema_20_1w_aligned[i]
        
        # LONG: KAMA uptrend + RSI momentum up + not ranging + weekly alignment
        long_signal = (kama_uptrend and 
                      rsi_momentum_up and 
                      not chop_ranging and 
                      weekly_uptrend)
        
        # SHORT: KAMA downtrend + RSI momentum down + not ranging + weekly alignment
        short_signal = (kama_downtrend and 
                       rsi_momentum_down and 
                       not chop_ranging and 
                       weekly_downtrend)
        
        # Exit conditions
        exit_long = position == 1 and (not kama_uptrend or chop_ranging)
        exit_short = position == -1 and (not kama_downtrend or chop_ranging)
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals