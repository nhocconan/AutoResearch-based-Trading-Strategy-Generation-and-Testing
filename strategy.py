#!/usr/bin/env python3
"""
1d_Weekly_RSI_Pullback_v1
Hypothesis: Combines weekly RSI extremes with daily pullbacks to the 20-day EMA.
Uses weekly trend filter to avoid counter-trend trades. Designed for low frequency
(10-25 trades/year) to work in both bull (buy dips in uptrend) and bear (sell rallies in downtrend)
markets by entering on pullbacks after extreme weekly RSI readings.
"""

name = "1d_Weekly_RSI_Pullback_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for trend filter and RSI
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Daily OHLCV
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- Weekly RSI (14) ---
    delta_w = pd.Series(df_1w['close']).diff()
    gain_w = delta_w.clip(lower=0)
    loss_w = -delta_w.clip(upper=0)
    avg_gain_w = gain_w.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss_w = loss_w.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs_w = avg_gain_w / avg_loss_w
    rsi_w = 100 - (100 / (1 + rs_w))
    rsi_w_values = rsi_w.values
    
    # --- Weekly EMA50 for trend filter ---
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # --- Daily EMA20 for pullback entries ---
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_w_values[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(ema_20[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Weekly RSI extremes
        rsi_overbought = rsi_w_values[i] > 70
        rsi_oversold = rsi_w_values[i] < 30
        
        # Weekly trend
        weekly_uptrend = close[i] > ema_50_1w_aligned[i]
        weekly_downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Daily pullback to EMA20
        pullback_to_ema = abs(close[i] - ema_20[i]) / ema_20[i] < 0.02  # Within 2% of EMA20
        
        if position == 0:
            # Long: weekly uptrend + weekly RSI oversold + pullback to EMA20
            if weekly_uptrend and rsi_oversold and pullback_to_ema:
                signals[i] = 0.25
                position = 1
            # Short: weekly downtrend + weekly RSI overbought + pullback to EMA20
            elif weekly_downtrend and rsi_overbought and pullback_to_ema:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        else:
            # Exit conditions: opposite weekly RSI extreme or trend change
            if position == 1:
                exit_signal = rsi_w_values[i] > 70 or not weekly_uptrend
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                exit_signal = rsi_w_values[i] < 30 or not weekly_downtrend
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals