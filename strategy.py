#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_ChopFilter_V3
Hypothesis: 1d strategy using Kaufman Adaptive Moving Average (KAMA) for trend direction, RSI(14) for momentum confirmation, and Choppiness Index(14) for regime filtering. Only takes longs when KAMA is rising, RSI>50, and CHOP<61.8 (trending regime); shorts when KAMA falling, RSI<50, and CHOP<61.8. Uses discrete position sizing (0.25) to minimize fee churn. Designed to work in both bull and bear markets by following adaptive trend with regime filter to avoid whipsaws in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need warmup for KAMA, RSI, and CHOP
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 for HTF trend filter (more stable than price alone)
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === KAMA Calculation (Kaufman Adaptive Moving Average) ===
    # ER = Efficiency Ratio = |net change| / sum of absolute changes
    # Smooth = ER * (fastest SC - slowest SC) + slowest SC
    # where SC = 2/(period+1)
    fast_sc = 2 / (2 + 1)   # for period=2
    slow_sc = 2 / (30 + 1)  # for period=30
    
    # Calculate change and volatility
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if False else None  # placeholder
    
    # Proper ER calculation
    change_seq = np.diff(close, prepend=close[0])
    abs_change = np.abs(change_seq)
    
    # Calculate 10-period ER
    net_change = np.abs(np.subtract(close[9:], close[:-9])) if len(close) >= 9 else np.array([])
    sum_abs_change = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if False else None
    
    # Vectorized ER calculation
    lookback = 10
    er = np.zeros(n)
    for i in range(lookback, n):
        net_ch = np.abs(close[i] - close[i-lookback])
        sum_abs = np.sum(np.abs(np.diff(close[i-lookback:i+1])))
        er[i] = net_ch / sum_abs if sum_abs != 0 else 0
    
    # Smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === RSI Calculation (14-period) ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (alpha = 1/period)
    lookback_rsi = 14
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    rsi = np.zeros(n)
    
    # Initial average
    if n > lookback_rsi:
        avg_gain[lookback_rsi] = np.mean(gain[1:lookback_rsi+1])
        avg_loss[lookback_rsi] = np.mean(loss[1:lookback_rsi+1])
        
        for i in range(lookback_rsi+1, n):
            avg_gain[i] = (avg_gain[i-1] * (lookback_rsi-1) + gain[i]) / lookback_rsi
            avg_loss[i] = (avg_loss[i-1] * (lookback_rsi-1) + loss[i]) / lookback_rsi
            rs = avg_gain[i] / avg_loss[i] if avg_loss[i] != 0 else 0
            rsi[i] = 100 - (100 / (1 + rs)) if avg_loss[i] != 0 else 50
    
    # === Choppiness Index Calculation (14-period) ===
    # CHOP = 100 * log10(sum(ATR1) / (n * log(n))) / log10(n)
    # where ATR1 = True Range
    tr1 = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr1[0] = high[0] - low[0]  # first TR
    
    atr_period = 14
    sum_tr = np.zeros(n)
    for i in range(atr_period, n):
        sum_tr[i] = np.sum(tr1[i-atr_period+1:i+1])
    
    chop = np.zeros(n)
    for i in range(atr_period, n):
        if sum_tr[i] > 0:
            log_n = np.log10(atr_period)
            chop[i] = 100 * np.log10(sum_tr[i] / (atr_period * log_n)) / log_n
        else:
            chop[i] = 50  # neutral
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 100 for indicators to stabilize)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Get current values
        close_val = close[i]
        kama_val = kama[i]
        rsi_val = rsi[i] if not np.isnan(rsi[i]) else 50
        chop_val = chop[i] if not np.isnan(chop[i]) else 50
        ema_val = ema_50_1w_aligned[i]
        
        # Skip if any data not ready
        if (np.isnan(kama_val) or np.isnan(rsi_val) or np.isnan(chop_val) or np.isnan(ema_val)):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Regime filter: only trade in trending markets (CHOP < 61.8)
        trending_regime = chop_val < 61.8
        
        # Long logic: KAMA rising, price above KAMA, RSI > 50, HTF uptrend, trending regime
        kama_rising = kama_val > kama[i-1] if i > 0 else False
        long_condition = (close_val > kama_val) and kama_rising and (rsi_val > 50) and (close_val > ema_val) and trending_regime
        
        # Short logic: KAMA falling, price below KAMA, RSI < 50, HTF downtrend, trending regime
        kama_falling = kama_val < kama[i-1] if i > 0 else False
        short_condition = (close_val < kama_val) and kama_falling and (rsi_val < 50) and (close_val < ema_val) and trending_regime
        
        # Exit logic: regime change to ranging OR opposite signal
        exit_long = (chop_val >= 61.8) or (close_val < kama_val) or (rsi_val < 40)
        exit_short = (chop_val >= 61.8) or (close_val > kama_val) or (rsi_val > 60)
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "1d_KAMA_Trend_RSI_ChopFilter_V3"
timeframe = "1d"
leverage = 1.0