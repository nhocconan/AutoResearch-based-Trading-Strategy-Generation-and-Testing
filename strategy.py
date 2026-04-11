#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Institutional Order Flow Detector using 1d Cumulative Delta and 1w Trend Filter
# Long when 1d cumulative delta (buy pressure) exceeds threshold and 1w trend is up
# Short when 1d cumulative delta (sell pressure) exceeds threshold and 1w trend is down
# Uses volume-weighted price momentum to detect institutional accumulation/distribution
# Designed for 20-40 trades/year on 6h timeframe with clear entry/exit rules

name = "6h_1d_1w_cumulative_delta_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 20 or len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate 1d cumulative delta (buy volume - sell volume approximation)
    # Using close price relative to high-low range as proxy for buying/selling pressure
    close_position = (df_1d['close'] - df_1d['low']) / (df_1d['high'] - df_1d['low'])
    close_position = close_position.fillna(0.5)  # Handle zero range
    delta_approx = (2 * close_position - 1) * df_1d['volume']  # -1 to +1 scaled by volume
    cumulative_delta = delta_approx.cumsum().values
    
    # Normalize cumulative delta by 20-period volatility for signal generation
    cum_delta_series = pd.Series(cumulative_delta)
    cum_delta_mean = cum_delta_series.rolling(window=20, min_periods=20).mean().values
    cum_delta_std = cum_delta_series.rolling(window=20, min_periods=20).std().values
    cum_delta_normalized = (cum_delta - cum_delta_mean) / cum_delta_std
    cum_delta_normalized = np.nan_to_num(cum_delta_normalized, nan=0.0)
    
    # Align normalized cumulative delta to 6h timeframe
    cum_delta_norm_aligned = align_htf_to_ltf(prices, df_1d, cum_delta_normalized)
    
    # Calculate 1w EMA(21) for trend filter
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after normalization period
        # Skip if any required data is invalid
        if (np.isnan(cum_delta_norm_aligned[i]) or np.isnan(ema_21_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1w trend direction
        is_uptrend = close[i] > ema_21_1w_aligned[i]
        is_downtrend = close[i] < ema_21_1w_aligned[i]
        
        # Entry conditions: significant cumulative delta with trend alignment
        # Long when buying pressure is strong and 1w trend up
        # Short when selling pressure is strong and 1w trend down
        cum_delta_long = cum_delta_norm_aligned[i] > 1.0 and is_uptrend
        cum_delta_short = cum_delta_norm_aligned[i] < -1.0 and is_downtrend
        
        # Exit conditions: pressure normalizes
        exit_long = cum_delta_norm_aligned[i] < 0.0  # Return to neutral
        exit_short = cum_delta_norm_aligned[i] > 0.0  # Return to neutral
        
        # Priority: entry > exit > hold
        if cum_delta_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif cum_delta_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals