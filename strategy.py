#!/usr/bin/env python3
# 6H_Keltner_Channel_Breakout_Momentum
# Hypothesis: Price breaking out of Keltner Channel (ATR-based) with momentum confirmation (RSI > 50 for longs, < 50 for shorts) and volume spike indicates institutional participation. Uses 1d trend filter (price above/below EMA50) to align with higher timeframe direction. Works in bull/bear by following trend and using volatility-based breakouts that capture momentum moves. Target: 15-30 trades/year per symbol.

name = "6H_Keltner_Channel_Breakout_Momentum"
timeframe = "6h"
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
    volume = prices['volume'].values
    
    # 6h indicators
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    volume_s = pd.Series(volume)
    
    # EMA20 for Keltner Channel middle
    ema20 = close_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Average True Range (ATR) for Keltner Channel width
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = np.zeros_like(tr)
    if len(tr) > 0:
        atr[0] = tr[0]
        for i in range(1, len(tr)):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14  # Wilder's smoothing
    # Prepend NaN for alignment (since we lost first bar)
    atr = np.concatenate([np.full(1, np.nan), atr])
    
    # Keltner Channel: Upper = EMA20 + 2*ATR, Lower = EMA20 - 2*ATR
    kc_upper = ema20 + 2 * atr
    kc_lower = ema20 - 2 * atr
    
    # RSI (14) for momentum confirmation
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    if len(gain) > 0:
        avg_gain[0] = gain[0] if len(gain) > 0 else 0
        avg_loss[0] = loss[0] if len(loss) > 0 else 0
        for i in range(1, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Prepend NaN for alignment (since we lost first bar in diff)
    rsi = np.concatenate([np.full(1, np.nan), rsi])
    
    # Volume average (20-period)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Daily trend filter: price above/below EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    daily_uptrend = close_1d > ema50_1d
    daily_downtrend = close_1d < ema50_1d
    
    # Align daily trend to 6h
    daily_uptrend_aligned = align_htf_to_ltf(prices, df_1d, daily_uptrend.astype(float))
    daily_downtrend_aligned = align_htf_to_ltf(prices, df_1d, daily_downtrend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema20[i]) or np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or
            np.isnan(rsi[i]) or np.isnan(vol_ma[i]) or
            np.isnan(daily_uptrend_aligned[i]) or np.isnan(daily_downtrend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        daily_up = daily_uptrend_aligned[i] > 0.5
        daily_down = daily_downtrend_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: price breaks above Keltner Upper + RSI > 50 (bullish momentum) + volume + daily uptrend
            if close[i] > kc_upper[i] and rsi[i] > 50 and volume_confirm and daily_up:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Keltner Lower + RSI < 50 (bearish momentum) + volume + daily downtrend
            elif close[i] < kc_lower[i] and rsi[i] < 50 and volume_confirm and daily_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price re-enters Keltner Channel or momentum fades
            if close[i] < kc_upper[i] or rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price re-enters Keltner Channel or momentum fades
            if close[i] > kc_lower[i] or rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals