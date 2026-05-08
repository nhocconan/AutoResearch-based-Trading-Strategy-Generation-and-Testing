#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum strategy with 4h trend filter and 1d volatility regime filter
# Uses 4h EMA(21) for trend direction and 1d ATR ratio (ATR(7)/ATR(30)) for volatility regime.
# Enters long when 4h EMA(21) is rising, 1h price > 1h EMA(13), and volatility regime is expanding (ATR ratio > 1.0).
# Enters short when 4h EMA(21) is falling, 1h price < 1h EMA(13), and volatility regime is expanding.
# Uses session filter (08-20 UTC) and fixed position size of 0.20 to control trade frequency.
# Designed to capture momentum bursts during volatile periods while avoiding choppy markets.
# Target: 15-35 trades/year per symbol.

name = "1h_EMA_Momentum_VolatilityRegime_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    
    # Get 1d data for volatility regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 4h EMA(21) for trend
    close_4h = df_4h['close'].values
    ema_4h = np.full(len(close_4h), np.nan)
    if len(close_4h) >= 21:
        ema_4h[20] = np.mean(close_4h[:21])
        for i in range(21, len(close_4h)):
            ema_4h[i] = (close_4h[i] * 2/22) + (ema_4h[i-1] * 20/22)
    
    # Calculate 1d ATR(7) and ATR(30) for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr_1d = np.zeros(len(close_1d))
    atr_7 = np.full(len(close_1d), np.nan)
    atr_30 = np.full(len(close_1d), np.nan)
    
    for i in range(len(close_1d)):
        if i == 0:
            tr_1d[i] = high_1d[i] - low_1d[i]
        else:
            tr_1d[i] = max(
                high_1d[i] - low_1d[i],
                abs(high_1d[i] - close_1d[i-1]),
                abs(low_1d[i] - close_1d[i-1])
            )
        
        if i >= 6:  # ATR(7)
            if i == 6:
                atr_7[i] = np.mean(tr_1d[:7])
            else:
                atr_7[i] = (atr_7[i-1] * 6 + tr_1d[i]) / 7
        
        if i >= 29:  # ATR(30)
            if i == 29:
                atr_30[i] = np.mean(tr_1d[:30])
            else:
                atr_30[i] = (atr_30[i-1] * 29 + tr_1d[i]) / 30
    
    # Calculate ATR ratio (ATR(7)/ATR(30)) for volatility regime
    atr_ratio = np.full(len(close_1d), np.nan)
    for i in range(len(close_1d)):
        if not np.isnan(atr_7[i]) and not np.isnan(atr_30[i]) and atr_30[i] != 0:
            atr_ratio[i] = atr_7[i] / atr_30[i]
    
    # Align 4h EMA(21) to 1h timeframe
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Align 1d ATR ratio to 1h timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Calculate 1h EMA(13) for entry timing
    ema_13 = np.full(n, np.nan)
    if n >= 13:
        ema_13[12] = np.mean(close[:13])
        for i in range(13, n):
            ema_13[i] = (close[i] * 2/14) + (ema_13[i-1] * 12/14)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 13)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(atr_ratio_aligned[i]) or 
            np.isnan(ema_13[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check conditions
        ema_4h_trend_up = ema_4h_aligned[i] > ema_4h_aligned[i-1]
        ema_4h_trend_down = ema_4h_aligned[i] < ema_4h_aligned[i-1]
        price_above_ema13 = close[i] > ema_13[i]
        price_below_ema13 = close[i] < ema_13[i]
        vol_expanding = atr_ratio_aligned[i] > 1.0
        
        if position == 0:
            # Look for entry: momentum in direction of 4h trend with volatility expansion
            if ema_4h_trend_up and price_above_ema13 and vol_expanding:
                signals[i] = 0.20
                position = 1
            elif ema_4h_trend_down and price_below_ema13 and vol_expanding:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: trend reversal or volatility contraction
            if (not ema_4h_trend_up) or (not vol_expanding):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: trend reversal or volatility contraction
            if (not ema_4h_trend_down) or (not vol_expanding):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals