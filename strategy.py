#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h session-based mean reversion with 4h trend filter and volume exhaustion
# Strategy trades mean reversion in ranging markets (8-20 UTC session) using Bollinger Bands
# 4h EMA50 defines trend direction - only take mean reversion trades in direction of 4h trend
# Volume exhaustion (volume < 0.5 x 20-period EMA) confirms low momentum environment for reversion
# Discrete position sizing (0.20) limits fee drag and drawdown
# Target: 60-150 total trades over 4 years (15-38/year) to avoid fee drag
# Works in bull/bear by following 4h trend - long in uptrends near BB lower, short in downtrends near BB upper

name = "1h_Session_MeanReversion_BB_VolumeExhaust_4hEMA50_Trend"
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
    
    # Session filter: 8-20 UTC (pre-compute for efficiency)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Bollinger Bands (20, 2.0) on 1h
    close_s = pd.Series(close)
    bb_middle = close_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2.0 * bb_std
    bb_lower = bb_middle - 2.0 * bb_std
    
    # Volume exhaustion: volume < 0.5 x 20-period EMA (low momentum environment)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_exhaustion = volume < (0.5 * vol_ema_20)
    
    # 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(bb_upper[i]) or 
            np.isnan(bb_lower[i]) or np.isnan(volume_exhaustion[i])):
            signals[i] = 0.0
            continue
        
        # Determine 4h trend bias
        uptrend = close[i] > ema_50_4h_aligned[i]
        downtrend = close[i] < ema_50_4h_aligned[i]
        
        if position == 0:  # Flat - look for mean reversion entries
            if in_session[i] and volume_exhaustion[i]:
                # Long: price at/below BB lower in uptrend (buy dip in uptrend)
                if close[i] <= bb_lower[i] and uptrend:
                    signals[i] = 0.20
                    position = 1
                # Short: price at/above BB upper in downtrend (sell rally in downtrend)
                elif close[i] >= bb_upper[i] and downtrend:
                    signals[i] = -0.20
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price returns to BB middle (mean reversion complete) OR session ends
            if close[i] >= bb_middle[i] or not in_session[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price returns to BB middle (mean reversion complete) OR session ends
            if close[i] <= bb_middle[i] or not in_session[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals