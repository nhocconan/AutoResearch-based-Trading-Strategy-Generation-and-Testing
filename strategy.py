#!/usr/bin/env python3
# 1H_4H1D_Trend_Filter_Momentum
# Hypothesis: Use 4h for primary trend direction (EMA50 slope) and 1d for regime filter (price vs EMA200).
# Enter on 1h momentum bursts (RSI > 60 for long, < 40 for short) only when aligned with 4h/1d trends.
# Exit when momentum fades (RSI returns to 40-60 range) or trend breaks.
# Designed for low trade frequency: ~20-40/year by requiring 3-timeframe alignment.
# Works in bull/bear by following higher timeframe trends and using momentum for timing.

name = "1H_4H1D_Trend_Filter_Momentum"
timeframe = "1h"
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
    
    # 1h RSI for momentum timing
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 4h trend: EMA50 slope (rising/falling)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_slope = np.diff(ema50_4h, prepend=ema50_4h[0])
    ema50_4h_rising = ema50_4h_slope > 0
    ema50_4h_falling = ema50_4h_slope < 0
    ema50_4h_rising_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h_rising.astype(float))
    ema50_4h_falling_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h_falling.astype(float))
    
    # 1d regime: price vs EMA200 (bull/bear filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    price_above_ema200 = close_1d > ema200_1d
    price_below_ema200 = close_1d < ema200_1d
    price_above_ema200_aligned = align_htf_to_ltf(prices, df_1d, price_above_ema200.astype(float))
    price_below_ema200_aligned = align_htf_to_ltf(prices, df_1d, price_below_ema200.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(rsi[i]) or 
            np.isnan(ema50_4h_rising_aligned[i]) or np.isnan(ema50_4h_falling_aligned[i]) or
            np.isnan(price_above_ema200_aligned[i]) or np.isnan(price_below_ema200_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        rsi_val = rsi[i]
        bull_momentum = rsi_val > 60
        bear_momentum = rsi_val < 40
        
        # 4h trend direction
        up4h = ema50_4h_rising_aligned[i] > 0.5
        down4h = ema50_4h_falling_aligned[i] > 0.5
        
        # 1d regime
        bull_regime = price_above_ema200_aligned[i] > 0.5
        bear_regime = price_below_ema200_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: 4h uptrend + 1d bull regime + bullish momentum
            if up4h and bull_regime and bull_momentum:
                signals[i] = 0.20
                position = 1
            # Enter short: 4h downtrend + 1d bear regime + bearish momentum
            elif down4h and bear_regime and bear_momentum:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: momentum fades (RSI < 50) or trend breaks
            if rsi_val < 50 or not (up4h and bull_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: momentum fades (RSI > 50) or trend breaks
            if rsi_val > 50 or not (down4h and bear_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals