#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_Momentum_Pullback_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h and 1d data once
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # === 4h EMA50 for trend filter ===
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # === 1d RSI14 for momentum filter ===
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi14_1d = 100 - (100 / (1 + rs))
    rsi14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi14_1d)
    
    # === 1h ATR14 for pullback filter ===
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # === 1h EMA20 for pullback entry ===
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # === Session filter: 08-20 UTC ===
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(rsi14_1d_aligned[i]) or
            np.isnan(atr14[i]) or np.isnan(ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: uptrend (price > 4h EMA50) + bullish momentum (RSI > 50) + pullback to 1h EMA20
            long_cond = (close[i] > ema50_4h_aligned[i] and
                        rsi14_1d_aligned[i] > 50 and
                        close[i] <= ema20[i] + 0.5 * atr14[i] and
                        close[i] >= ema20[i] - 0.5 * atr14[i])
            
            # Short: downtrend (price < 4h EMA50) + bearish momentum (RSI < 50) + pullback to 1h EMA20
            short_cond = (close[i] < ema50_4h_aligned[i] and
                         rsi14_1d_aligned[i] < 50 and
                         close[i] >= ema20[i] - 0.5 * atr14[i] and
                         close[i] <= ema20[i] + 0.5 * atr14[i])
            
            if long_cond:
                signals[i] = 0.20
                position = 1
            elif short_cond:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: break of 4h EMA50 or RSI < 40
            exit_cond = (close[i] < ema50_4h_aligned[i] or
                        rsi14_1d_aligned[i] < 40)
            if exit_cond:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: break of 4h EMA50 or RSI > 60
            exit_cond = (close[i] > ema50_4h_aligned[i] or
                        rsi14_1d_aligned[i] > 60)
            if exit_cond:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

# Hypothesis: 1h momentum pullback strategy using 4h EMA50 for trend direction and 1d RSI14 for momentum filter.
# Enters on pullbacks to 1h EMA20 within 0.5x ATR band when trend and momentum align.
# Exits when trend breaks (price crosses 4h EMA50) or momentum fades (RSI extremes).
# Uses session filter (08-20 UTC) to avoid low-liquidity hours.
# Designed for 15-30 trades/year (~60-120 over 4 years) to minimize fee drag.
# Works in bull markets (trend following) and bear markets (counter-trend pulls in downtrends).