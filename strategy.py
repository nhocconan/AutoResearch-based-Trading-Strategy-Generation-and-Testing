#!/usr/bin/env python3
"""
1d_KAMA_RSI_ChopFilter_v2
Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction,
RSI(14) for momentum confirmation, and Choppiness Index(14) as regime filter to avoid whipsaws.
Long when KAMA rising, RSI > 50, and CHOP > 61.8 (ranging market) for mean reversion longs at support.
Short when KAMA falling, RSI < 50, and CHOP > 61.8 for mean reversion shorts at resistance.
Uses weekly trend filter: only trade in direction of weekly EMA20 to avoid counter-trend whipsaws in strong trends.
Designed for low trade frequency (7-25/year) with discrete position sizing (0.25) to minimize fee drag.
Works in both bull and bear markets by adapting to ranging regimes while respecting weekly trend.
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # KAMA calculation (ER = 10, fast = 2, slow = 30)
    close_s = pd.Series(close)
    direction = np.abs(close_s.diff(10)).values
    volatility = close_s.diff().abs().rolling(window=10, min_periods=10).sum().values
    er = np.where(volatility > 0, direction / volatility, 0)
    sc = (er * (2/2 - 1/30) + 1/30) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = close_s.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index(14)
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).sum().values
    hh = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max().values
    ll = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min().values
    chop = np.where((hh - ll) > 0, -100 * np.log10(atr / (hh - ll)) / np.log10(atr_period), 50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of weekly EMA (20), KAMA (10), RSI (14), CHOP (14)
    start_idx = max(20, 10, 14, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(chop[i])):
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
            continue
        
        ema_20_1w_val = ema_20_1w_aligned[i]
        kama_val = kama[i]
        kama_prev = kama[i-1]
        rsi_val = rsi[i]
        chop_val = chop[i]
        close_val = close[i]
        
        if position == 0:
            # KAMA rising = bullish bias, falling = bearish bias
            kama_rising = kama_val > kama_prev
            kama_falling = kama_val < kama_prev
            
            # Long: KAMA rising, RSI > 50, choppy market (mean reversion long at support)
            long_signal = kama_rising and (rsi_val > 50) and (chop_val > 61.8)
            # Short: KAMA falling, RSI < 50, choppy market (mean reversion short at resistance)
            short_signal = kama_falling and (rsi_val < 50) and (chop_val > 61.8)
            
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
            # Exit: KAMA falling (trend change) OR chop < 38.2 (trending market - follow weekly trend)
            if (not kama_rising) or (chop_val < 38.2 and close_val < ema_20_1w_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: KAMA rising (trend change) OR chop < 38.2 (trending market - follow weekly trend)
            if (not kama_falling) or (chop_val < 38.2 and close_val > ema_20_1w_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_RSI_ChopFilter_v2"
timeframe = "1d"
leverage = 1.0