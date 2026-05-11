#!/usr/bin/env python3
# 6h_WeeklyPivot_Reversal_Trend
# Hypothesis: Uses weekly pivot points to identify key support/resistance levels. 
# Long when price rejects weekly S1/S2 with bullish engulfing candle and weekly trend is up.
# Short when price rejects weekly R1/R2 with bearish engulfing candle and weekly trend is down.
# Weekly trend defined by price above/below weekly 20-period EMA. 
# Designed to work in both bull and bear markets by fading extremes in ranging conditions 
# and continuing trends when price breaks pivot levels with momentum.
# Weekly pivots provide institutional reference points; engulfing candles confirm rejection/breakout.

name = "6h_WeeklyPivot_Reversal_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for pivot points and trend
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 20:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    open_price = prices['open'].values
    
    # --- Weekly pivot points (standard formula) ---
    # P = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    H_w = df_w['high'].values
    L_w = df_w['low'].values
    C_w = df_w['close'].values
    
    P_w = (H_w + L_w + C_w) / 3.0
    R1_w = 2 * P_w - L_w
    S1_w = 2 * P_w - H_w
    R2_w = P_w + (H_w - L_w)
    S2_w = P_w - (H_w - L_w)
    R3_w = H_w + 2 * (P_w - L_w)
    S3_w = L_w - 2 * (H_w - P_w)
    
    # Weekly trend: price above/below 20 EMA
    close_w_series = pd.Series(C_w)
    ema20_w = close_w_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    w_uptrend = C_w > ema20_w
    w_downtrend = C_w < ema20_w
    
    # Align weekly data to 6h timeframe
    P_w_a = align_htf_to_ltf(prices, df_w, P_w)
    R1_w_a = align_htf_to_ltf(prices, df_w, R1_w)
    S1_w_a = align_htf_to_ltf(prices, df_w, S1_w)
    R2_w_a = align_htf_to_ltf(prices, df_w, R2_w)
    S2_w_a = align_htf_to_ltf(prices, df_w, S2_w)
    R3_w_a = align_htf_to_ltf(prices, df_w, R3_w)
    S3_w_a = align_htf_to_ltf(prices, df_w, S3_w)
    w_uptrend_a = align_htf_to_ltf(prices, df_w, w_uptrend)
    w_downtrend_a = align_htf_to_ltf(prices, df_w, w_downtrend)
    
    # Bullish engulfing: current green candle engulfs previous red candle
    bull_engulf = (close > open_price) & (open_price < np.roll(close, 1)) & (close > np.roll(open_price, 1))
    # Bearish engulfing: current red candle engulfs previous green candle
    bear_engulf = (close < open_price) & (open_price > np.roll(close, 1)) & (close < np.roll(open_price, 1))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 2 for engulfing, 20 for weekly EMA
    start_idx = max(2, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(P_w_a[i]) or np.isnan(R1_w_a[i]) or np.isnan(S1_w_a[i]) or
            np.isnan(R2_w_a[i]) or np.isnan(S2_w_a[i]) or np.isnan(R3_w_a[i]) or
            np.isnan(S3_w_a[i]) or np.isnan(w_uptrend_a[i]) or np.isnan(w_downtrend_a[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price rejects S1/S2 with bullish engulfing and weekly uptrend
            if (low[i] <= S1_w_a[i] or low[i] <= S2_w_a[i]) and bull_engulf[i] and w_uptrend_a[i]:
                signals[i] = 0.25
                position = 1
            # Short: price rejects R1/R2 with bearish engulfing and weekly downtrend
            elif (high[i] >= R1_w_a[i] or high[i] >= R2_w_a[i]) and bear_engulf[i] and w_downtrend_a[i]:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price breaks above R1 (momentum) OR weekly trend turns down
                if high[i] >= R1_w_a[i] or not w_uptrend_a[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price breaks below S1 (momentum) OR weekly trend turns up
                if low[i] <= S1_w_a[i] or not w_downtrend_a[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals