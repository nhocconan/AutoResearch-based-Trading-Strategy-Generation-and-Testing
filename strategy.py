#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Bollinger Band mean reversion with 4h trend filter and session filter.
# Long when price touches lower BB(20,2) AND 4h EMA50 uptrend (price > EMA50) AND UTC 08-20 session.
# Short when price touches upper BB(20,2) AND 4h EMA50 downtrend (price < EMA50) AND UTC 08-20 session.
# Uses discrete position size 0.20. Bollinger Bands identify overextended moves, 4h EMA50 ensures alignment with higher timeframe trend.
# Session filter reduces noise trades outside active hours. Designed to work in both bull (buy dips) and bear (sell rallies) markets.
# Target: 80-150 trades over 4 years (20-37/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === 1h Indicators: Bollinger Bands (20,2) ===
    close_s = pd.Series(close)
    bb_ma = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_ma + 2 * bb_std
    bb_lower = bb_ma - 2 * bb_std
    
    # Get 4h data once before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:  # Need enough for EMA50 calculation
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # === 4h Indicators: EMA50 for trend filter ===
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h EMA50 to 1h timeframe
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Session filter: UTC 08-20 (active trading hours)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for EMA, 20 for BB)
    warmup = 60
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(ema_50_4h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        bb_up = bb_upper[i]
        bb_low = bb_lower[i]
        ema_4h = ema_50_4h_aligned[i]
        in_session = (8 <= hours[i] <= 20)
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price returns to middle band or session ends
            if price >= bb_ma[i] or not in_session:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price returns to middle band or session ends
            if price <= bb_ma[i] or not in_session:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0 and in_session:
            # LONG: price touches lower BB AND 4h EMA50 uptrend (price > EMA50)
            if price <= bb_low and price > ema_4h:
                signals[i] = 0.20
                position = 1
            
            # SHORT: price touches upper BB AND 4h EMA50 downtrend (price < EMA50)
            elif price >= bb_up and price < ema_4h:
                signals[i] = -0.20
                position = -1
        
        else:
            signals[i] = position * 0.20
    
    return signals

name = "1h_BB_MeanReversion_4hEMA50_SessionFilter_V1"
timeframe = "1h"
leverage = 1.0