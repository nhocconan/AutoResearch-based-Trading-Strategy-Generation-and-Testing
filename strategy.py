#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 1d Bollinger Band squeeze (volatility contraction) + 4h trend filter
# Bollinger Band squeeze indicates low volatility, often preceding breakouts
# Trend filter ensures we only trade in direction of higher timeframe trend
# Session filter (08-20 UTC) reduces noise outside active trading hours
# Target: 15-30 trades/year (~60-120 total over 4 years) with disciplined entries

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # === 1d Bollinger Bands (20, 2) for squeeze detection ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Bollinger Bands
    bb_length = 20
    bb_mult = 2.0
    
    # Basis (SMA)
    basis = np.full_like(close_1d, np.nan)
    for i in range(len(close_1d)):
        if i >= bb_length - 1:
            basis[i] = np.mean(close_1d[i - bb_length + 1:i + 1])
        elif i > 0:
            basis[i] = np.mean(close_1d[0:i + 1])
        else:
            basis[i] = close_1d[0]
    
    # Standard deviation
    bb_dev = np.full_like(close_1d, np.nan)
    for i in range(len(close_1d)):
        if i >= bb_length - 1:
            bb_dev[i] = np.std(close_1d[i - bb_length + 1:i + 1])
        elif i > 0:
            bb_dev[i] = np.std(close_1d[0:i + 1])
        else:
            bb_dev[i] = 0.0
    
    # Upper and lower bands
    upper = basis + bb_mult * bb_dev
    lower = basis - bb_mult * bb_dev
    
    # Bollinger Band width (normalized)
    bb_width = np.full_like(close_1d, np.nan)
    for i in range(len(close_1d)):
        if basis[i] != 0:
            bb_width[i] = (upper[i] - lower[i]) / basis[i]
        else:
            bb_width[i] = 0.0
    
    # Bollinger Band squeeze: width below 20-period percentile (20th percentile)
    bb_width_percentile = np.full_like(bb_width, np.nan)
    lookback = 20
    for i in range(len(bb_width)):
        if i >= lookback - 1:
            window = bb_width[i - lookback + 1:i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) > 0:
                bb_width_percentile[i] = np.percentile(valid, 20)
            else:
                bb_width_percentile[i] = 0.0
        elif i > 0:
            window = bb_width[0:i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) > 0:
                bb_width_percentile[i] = np.percentile(valid, 20)
            else:
                bb_width_percentile[i] = 0.0
        else:
            bb_width_percentile[i] = bb_width[0] if not np.isnan(bb_width[0]) else 0.0
    
    # Squeeze condition: current width <= 20th percentile width
    squeeze = bb_width <= bb_width_percentile
    
    # === 4h EMA(34) for trend filter ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate EMA(34)
    ema_len = 34
    ema_4h = np.full_like(close_4h, np.nan)
    if len(close_4h) >= ema_len:
        ema_4h[ema_len - 1] = np.mean(close_4h[:ema_len])  # seed
        alpha = 2 / (ema_len + 1)
        for i in range(ema_len, len(close_4h)):
            ema_4h[i] = alpha * close_4h[i] + (1 - alpha) * ema_4h[i - 1]
    else:
        for i in range(len(close_4h)):
            ema_4h[i] = np.mean(close_4h[0:i + 1]) if i >= 0 else close_4h[0]
    
    # Align 1d squeeze and 4h EMA to 1h timeframe
    squeeze_aligned = align_htf_to_ltf(prices, df_1d, squeeze)
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # === 1h Donchian Channel (20) for entry timing ===
    donch_len = 20
    donch_high = np.full_like(high, np.nan)
    donch_low = np.full_like(low, np.nan)
    
    for i in range(len(high)):
        if i >= donch_len - 1:
            donch_high[i] = np.max(high[i - donch_len + 1:i + 1])
            donch_low[i] = np.min(low[i - donch_len + 1:i + 1])
        elif i > 0:
            donch_high[i] = np.max(high[0:i + 1])
            donch_low[i] = np.min(low[0:i + 1])
        else:
            donch_high[i] = high[0]
            donch_low[i] = low[0]
    
    # Signals
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup period
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(squeeze_aligned[i]) or 
            np.isnan(ema_4h_aligned[i]) or 
            np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or
            not in_session[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat and in squeeze
        if position == 0 and squeeze_aligned[i]:
            # Long: price breaks above Donchian high AND above 4h EMA (uptrend)
            if close[i] > donch_high[i] and close[i] > ema_4h_aligned[i]:
                signals[i] = 0.20
                position = 1
                continue
            # Short: price breaks below Donchian low AND below 4h EMA (downtrend)
            elif close[i] < donch_low[i] and close[i] < ema_4h_aligned[i]:
                signals[i] = -0.20
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price breaks below Donchian low OR squeeze ends (volatility expansion)
            if close[i] < donch_low[i] or not squeeze_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price breaks above Donchian high OR squeeze ends
            if close[i] > donch_high[i] or not squeeze_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_BBSqueeze_DonchianTrend"
timeframe = "1h"
leverage = 1.0