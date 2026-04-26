#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_ChopFilter_v1
Hypothesis: 1d timeframe strategy using KAMA for trend direction, RSI(14) for momentum confirmation, and Choppiness Index (CHOP) as regime filter. Long when KAMA up, RSI>50, CHOP<61.8 (trending); Short when KAMA down, RSI<50, CHOP<61.8. Uses 1w HTF for higher-timeframe trend alignment. Discrete sizing 0.25. Target 10-20 trades/year to minimize fee drag while capturing sustained trends in both bull and bear markets.
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
    open_time = prices['open_time'].values
    
    # Session filter: UTC 8-20 for institutional activity (applies to 1d bars via index hour)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for primary indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # === 1d Indicators ===
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # KAMA (Kaufman Adaptive Moving Average) - trend direction
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_1d, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=1)  # sum of |close[t] - close[t-1]| over 10 periods
    # Pad volatility to match length
    volatility = np.concatenate([np.full(9, np.nan), volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants: fastest EMA=2, slowest EMA=30
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # (ER*(0.5-0.0667)+0.0667)^2
    # Initialize KAMA
    kama = np.full_like(close_1d, np.nan)
    kama[9] = close_1d[9]  # seed at period 10
    for i in range(10, len(close_1d)):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # RSI(14) - momentum
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    # Rolling average
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Pad RSI to align with close_1d (first 14 values NaN)
    rsi = np.concatenate([np.full(14, np.nan), rsi])
    
    # Choppiness Index (CHOP) - regime filter
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    # Sum of TR over 14 periods
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    # CHOP = 100 * log10(sum_tr / (hh - ll)) / log10(14)
    range_hl = hh - ll
    chop = np.where(range_hl > 0, 100 * np.log10(sum_tr / range_hl) / np.log10(14), 50)
    # Pad CHOP (first 14 values NaN)
    chop = np.concatenate([np.full(14, np.nan), chop])
    
    # === 1w HTF Trend Filter ===
    close_1w = df_1w['close'].values
    # Simple 1w EMA20 for trend
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    # Align to 1d timeframe
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Align 1d indicators to 4h? No - we are on 1d timeframe, so we need to align to 1d bars
    # But our prices are 4h? Wait: timeframe="1d" means we expect daily bars
    # However, the engine may still pass 4h data? No - timeframe declares the expected resolution
    # We must align our HTF data to the prices timeframe (which should be 1d if timeframe="1d")
    # But to be safe, we align to the prices index regardless
    
    # Align all 1d indicators to prices timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of KAMA seed (10), RSI (14), CHOP (14), EMA20_1w
    start_idx = max(10, 14, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or
            not in_session[i]):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        chop_val = chop_aligned[i]
        ema_20_1w_val = ema_20_1w_aligned[i]
        close_val = close[i]
        
        # Regime filter: only trade when market is trending (CHOP < 61.8)
        is_trending = chop_val < 61.8
        
        if position == 0:
            # Long: KAMA up (price > KAMA), RSI > 50, trending regime, HTF uptrend (close > EMA20_1w)
            long_signal = (close_val > kama_val) and (rsi_val > 50) and is_trending and (close_val > ema_20_1w_val)
            # Short: KAMA down (price < KAMA), RSI < 50, trending regime, HTF downtrend (close < EMA20_1w)
            short_signal = (close_val < kama_val) and (rsi_val < 50) and is_trending and (close_val < ema_20_1w_val)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: trend reversal (price < KAMA) OR RSI < 40 (momentum loss) OR HTF trend change (close < EMA20_1w)
            if (close_val < kama_val) or (rsi_val < 40) or (close_val < ema_20_1w_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: trend reversal (price > KAMA) OR RSI > 60 (momentum loss) OR HTF trend change (close > EMA20_1w)
            if (close_val > kama_val) or (rsi_val > 60) or (close_val > ema_20_1w_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Direction_RSI_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0