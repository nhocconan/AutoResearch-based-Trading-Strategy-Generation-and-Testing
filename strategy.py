#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d 12-hour EMA trend with weekly Bollinger Band squeeze breakout
# Uses weekly Bollinger Band width percentile to detect low volatility regimes (squeeze)
# and breaks out in the direction of the 12-hour EMA trend. This captures explosive moves
# after consolidation periods, working in both bull and bear markets by following the
# intermediate-term trend. Target: 15-25 trades/year to minimize fee decay while capturing
# significant momentum bursts. Focus on BTC/ETH as primary assets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Bollinger Bands
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Get 12h data for EMA trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 12:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    n_1w = len(close_1w)
    
    # Weekly Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2.0
    sma_1w = np.full(n_1w, np.nan)
    std_1w = np.full(n_1w, np.nan)
    upper_bb = np.full(n_1w, np.nan)
    lower_bb = np.full(n_1w, np.nan)
    bb_width = np.full(n_1w, np.nan)
    
    for i in range(bb_period, n_1w):
        sma_1w[i] = np.mean(close_1w[i-bb_period:i])
        std_1w[i] = np.std(close_1w[i-bb_period:i])
        upper_bb[i] = sma_1w[i] + bb_std * std_1w[i]
        lower_bb[i] = sma_1w[i] - bb_std * std_1w[i]
        bb_width[i] = upper_bb[i] - lower_bb[i]
    
    # Calculate BB width percentile (252-week lookback for regime)
    lookback = min(252, n_1w)
    bb_width_percentile = np.full(n_1w, np.nan)
    for i in range(lookback, n_1w):
        window = bb_width[i-lookback:i]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            bb_width_percentile[i] = (np.sum(valid <= bb_width[i]) / len(valid)) * 100
    
    # 12-hour EMA (12-period)
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=12, adjust=False).values
    
    # Align indicators to 1d
    bb_width_percentile_aligned = align_htf_to_ltf(prices, df_1w, bb_width_percentile)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup period
    start_idx = 30  # Need enough data for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(bb_width_percentile_aligned[i]) or 
            np.isnan(ema_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Squeeze condition: BB width in lowest 10% percentile (low volatility)
        squeeze = bb_width_percentile_aligned[i] < 10
        
        # Trend direction from 12h EMA
        uptrend = price > ema_12h_aligned[i]
        downtrend = price < ema_12h_aligned[i]
        
        if position == 0:
            # Enter long on squeeze breakout in uptrend
            if squeeze and uptrend:
                signals[i] = size
                position = 1
            # Enter short on squeeze breakout in downtrend
            elif squeeze and downtrend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long when trend reverses or squeeze ends
            if not uptrend or not squeeze:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short when trend reverses or squeeze ends
            if not downtrend or not squeeze:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_12hEMA_WeeklyBB_Squeeze"
timeframe = "1d"
leverage = 1.0