#!/usr/bin/env python3
# Hypothesis: 1d Bollinger Bands squeeze breakout with 1w trend filter and volume confirmation.
# Uses Bollinger Bands width percentile to detect low volatility squeeze (breakout setup).
# Enters on breakout above upper BB or below lower BB when 1w EMA(20) trend is aligned.
# Volume confirmation requires >1.5x 20-period average. Designed for 1d timeframe
# with ~30-80 total trades over 4 years to minimize fee drift.
# Works in both bull and breakout markets by capturing volatility expansion after consolidation.

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
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_stddev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma + (bb_std * bb_stddev)
    lower_band = sma - (bb_std * bb_stddev)
    bb_width = upper_band - lower_band
    
    # Bollinger Bands width percentile (50-period lookback) to detect squeeze
    bb_width_ma = pd.Series(bb_width).rolling(window=50, min_periods=50).mean().values
    bb_width_std = pd.Series(bb_width).rolling(window=50, min_periods=50).std().values
    bb_width_zscore = (bb_width - bb_width_ma) / bb_width_std
    # Squeeze when BB width is significantly below average (z-score < -1.0)
    squeeze = bb_width_zscore < -1.0
    
    # Volume filter: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    
    start_idx = max(bb_period, 50, 20)  # Wait for BB, width percentile, and 1w EMA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(sma[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(bb_width_zscore[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        long_breakout = close[i] > upper_band[i]
        short_breakout = close[i] < lower_band[i]
        
        # Trend filter: price relative to 1w EMA(20)
        uptrend = close[i] > ema_20_1w_aligned[i]
        downtrend = close[i] < ema_20_1w_aligned[i]
        
        # Entry conditions: breakout + squeeze + volume + trend alignment
        long_entry = long_breakout and squeeze[i] and volume_confirm[i] and uptrend
        short_entry = short_breakout and squeeze[i] and volume_confirm[i] and downtrend
        
        # Exit conditions: mean reversion to middle band or opposite breakout
        long_exit = close[i] < sma[i]  # Return to middle band
        short_exit = close[i] > sma[i]  # Return to middle band
        
        # Generate signals
        if long_entry:
            signals[i] = 0.25
        elif short_entry:
            signals[i] = -0.25
        elif long_exit and (i > 0 and signals[i-1] > 0):
            signals[i] = 0.0
        elif short_exit and (i > 0 and signals[i-1] < 0):
            signals[i] = 0.0
        else:
            # Hold previous signal
            signals[i] = signals[i-1] if i > 0 else 0.0
    
    return signals

name = "1d_BollingerSqueeze_Breakout_1wEMA20_VolumeConfirm"
timeframe = "1d"
leverage = 1.0