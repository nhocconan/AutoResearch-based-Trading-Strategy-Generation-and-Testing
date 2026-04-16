#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Supertrend for trend direction and 1d RSI(2) for mean reversion timing.
# Long when 4h Supertrend is bullish AND 1d RSI(2) < 10 (oversold) AND price > 1h EMA(50) (pullback entry).
# Short when 4h Supertrend is bearish AND 1d RSI(2) > 90 (overbought) AND price < 1h EMA(50) (pullback entry).
# Exit when 1d RSI(2) crosses 50 (mean reversion complete) or Supertrend flips.
# Uses discrete position size 0.20. Session filter: 08-20 UTC to avoid low-volume hours.
# Target: 60-150 total trades over 4 years (15-37/year) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data once before loop for Supertrend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # === 4h Indicators: Supertrend (ATR=10, mult=3.0) ===
    # True Range
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(10)
    atr = pd.Series(tr).ewm(alpha=1/10, adjust=False, min_periods=10).mean().values
    
    # Supertrend calculation
    hl2 = (high_4h + low_4h) / 2.0
    upper = hl2 + 3.0 * atr
    lower = hl2 - 3.0 * atr
    
    supertrend = np.zeros_like(close_4h)
    direction = np.ones_like(close_4h)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upper[0]
    direction[0] = 1
    
    for i in range(1, len(close_4h)):
        if close_4h[i] > supertrend[i-1]:
            direction[i] = 1
        elif close_4h[i] < supertrend[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
        
        if direction[i] == 1:
            supertrend[i] = max(lower[i], supertrend[i-1])
        else:
            supertrend[i] = min(upper[i], supertrend[i-1])
    
    # Align 4h Supertrend direction to 1h timeframe
    supertrend_dir_aligned = align_htf_to_ltf(prices, df_4h, direction)
    
    # Get 1d data once before loop for RSI(2)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: RSI(2) for mean reversion timing ===
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    # Wilder's smoothing for RSI
    avg_gain = pd.Series(gain).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Align 1d RSI(2) to 1h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # 1h EMA(50) for pullback entry timing
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    for i in range(warmup, n):
        # Session filter: only trade between 08:00 and 20:00 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(supertrend_dir_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(ema_50[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values
        std = supertrend_dir_aligned[i]
        rsi_val = rsi_aligned[i]
        price = close[i]
        ema = ema_50[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if RSI(2) crosses above 50 (mean reversion) or Supertrend turns bearish
            if rsi_val >= 50 or std == -1:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if RSI(2) crosses below 50 (mean reversion) or Supertrend turns bullish
            if rsi_val <= 50 or std == 1:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: 4h Supertrend bullish AND 1d RSI(2) < 10 (oversold) AND price > 1h EMA(50)
            if (std == 1) and (rsi_val < 10) and (price > ema):
                signals[i] = 0.20
                position = 1
                entry_price = price
            
            # SHORT: 4h Supertrend bearish AND 1d RSI(2) > 90 (overbought) AND price < 1h EMA(50)
            elif (std == -1) and (rsi_val > 90) and (price < ema):
                signals[i] = -0.20
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.20
    
    return signals

name = "1h_4hSupertrend_1dRSI2_EMA50_Pullback_V1"
timeframe = "1h"
leverage = 1.0