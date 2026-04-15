#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Supertrend for direction + 1h RSI pullback for entry + volume confirmation + session filter (08-20 UTC).
# Long when: 4h Supertrend uptrend + 1h RSI < 30 (pullback) + volume > 1.5x 20-period avg + in session.
# Short when: 4h Supertrend downtrend + 1h RSI > 70 (pullback) + volume > 1.5x 20-period avg + in session.
# Uses discrete position sizing (0.20) to minimize fee churn. Target: 15-30 trades/year.
# Supertrend filters choppy markets, RSI captures mean-reversion in trends, volume confirms conviction.
# Works in bull markets (buy pullbacks in uptrend) and bear markets (sell rallies in downtrend).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # === 4h Indicator: Supertrend (direction filter) ===
    atr_period = 10
    atr_multiplier = 3.0
    
    # True Range
    high_shift = np.roll(df_4h['high'], 1)
    low_shift = np.roll(df_4h['low'], 1)
    close_shift = np.roll(df_4h['close'], 1)
    high_shift[0] = df_4h['high'].iloc[0]
    low_shift[0] = df_4h['low'].iloc[0]
    close_shift[0] = df_4h['close'].iloc[0]
    
    tr1 = df_4h['high'] - df_4h['low']
    tr2 = np.abs(df_4h['high'] - close_shift)
    tr3 = np.abs(df_4h['low'] - close_shift)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR using Wilder's smoothing
    atr_4h = np.zeros_like(tr)
    atr_4h[atr_period-1] = np.mean(tr[:atr_period])
    for i in range(atr_period, len(tr)):
        atr_4h[i] = (atr_4h[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Supertrend calculation
    hl2 = (df_4h['high'] + df_4h['low']) / 2
    upper_band = hl2 + (atr_multiplier * atr_4h)
    lower_band = hl2 - (atr_multiplier * atr_4h)
    
    supertrend = np.zeros_like(close)
    direction = np.ones_like(close)  # 1 for uptrend, -1 for downtrend
    
    supertrend[atr_period-1] = upper_band[atr_period-1]
    direction[atr_period-1] = 1
    
    for i in range(atr_period, len(close)):
        prev_close = df_4h['close'].iloc[i-1]
        prev_supertrend = supertrend[i-1]
        prev_direction = direction[i-1]
        
        if prev_close > prev_supertrend:
            supertrend[i] = max(lower_band[i], prev_supertrend)
            direction[i] = 1
        else:
            supertrend[i] = min(upper_band[i], prev_supertrend)
            direction[i] = -1
    
    supertrend_4h_aligned = align_htf_to_ltf(prices, df_4h, supertrend)
    direction_4h_aligned = align_htf_to_ltf(prices, df_4h, direction)
    
    # 1h RSI (pullback entry)
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[rsi_period-1] = np.mean(gain[:rsi_period])
    avg_loss[rsi_period-1] = np.mean(loss[:rsi_period])
    
    for i in range(rsi_period, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
        avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(atr_period*2, rsi_period*2, 20) + 10
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(supertrend_4h_aligned[i]) or np.isnan(direction_4h_aligned[i]) or
            np.isnan(rsi[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. 4h Supertrend uptrend (direction == 1)
        # 2. 1h RSI < 30 (pullback/oversold)
        # 3. Volume confirmation
        if (direction_4h_aligned[i] == 1) and \
           (rsi[i] < 30) and vol_confirm:
            signals[i] = 0.20
        
        # === SHORT CONDITIONS ===
        # 1. 4h Supertrend downtrend (direction == -1)
        # 2. 1h RSI > 70 (pullback/overbought)
        # 3. Volume confirmation
        elif (direction_4h_aligned[i] == -1) and \
             (rsi[i] > 70) and vol_confirm:
            signals[i] = -0.20
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1h_Supertrend4h_RSIPullback_Volume_v1"
timeframe = "1h"
leverage = 1.0