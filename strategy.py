#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Squeeze Breakout with 12h ATR Trend Filter and Volume Confirmation
# Uses Bollinger Bands (20,2.0) to identify low volatility squeezes
# Breakout direction filtered by 12h ATR-based trend (price > EMA20 + 0.5*ATR = uptrend)
# Volume spike (1.5x 20-period average) confirms breakout validity
# Works in bull markets via buying upward breakouts in uptrends and bear markets via selling downward breakouts in downtrends
# Discrete sizing 0.25 minimizes fee churn. Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_Bollinger_Squeeze_Breakout_12hATRTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 12h data ONCE before loop (MTF Rule #1)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h EMA20 and ATR14 for trend filter
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    ema_20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    # ATR calculation: max(high-low, abs(high-prev_close), abs(low-prev_close))
    prev_close_12h = np.roll(close_12h, 1)
    prev_close_12h[0] = close_12h[0]
    tr_12h = np.maximum(high_12h - low_12h, np.maximum(np.abs(high_12h - prev_close_12h), np.abs(low_12h - prev_close_12h)))
    atr_14_12h = pd.Series(tr_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Trend filter: price > EMA20 + 0.5*ATR = uptrend, price < EMA20 - 0.5*ATR = downtrend
    trend_up_12h = ema_20_12h + 0.5 * atr_14_12h
    trend_down_12h = ema_20_12h - 0.5 * atr_14_12h
    trend_up_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_up_12h)
    trend_down_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_down_12h)
    
    # Bollinger Bands (20, 2.0) on 6h
    bb_period = 20
    bb_std = 2.0
    sma_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    bb_upper = sma_20 + (bb_std * std_20)
    bb_lower = sma_20 - (bb_std * std_20)
    bb_width = bb_upper - bb_lower
    
    # Bollinger Band Squeeze: width < 20-period average width (low volatility)
    bb_width_ma_20 = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    bb_squeeze = bb_width < bb_width_ma_20
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, bb_period, 20)  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(sma_20[i]) or np.isnan(std_20[i]) or np.isnan(bb_width_ma_20[i]) or
            np.isnan(trend_up_12h_aligned[i]) or np.isnan(trend_down_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_bb_upper = bb_upper[i]
        curr_bb_lower = bb_lower[i]
        curr_bb_squeeze = bb_squeeze[i]
        curr_volume_spike = volume_spike[i]
        curr_trend_up = trend_up_12h_aligned[i]
        curr_trend_down = trend_down_12h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require Bollinger Band squeeze and volume spike
            if curr_bb_squeeze and curr_volume_spike:
                # Bullish entry: price breaks above upper BB AND above 12h uptrend level
                if curr_close > curr_bb_upper and curr_close > curr_trend_up:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: price breaks below lower BB AND below 12h downtrend level
                elif curr_close < curr_bb_lower and curr_close < curr_trend_down:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when price falls below middle BB (SMA20) or below lower BB
            if curr_close < sma_20[i] or curr_close < curr_bb_lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price rises above middle BB (SMA20) or above upper BB
            if curr_close > sma_20[i] or curr_close > curr_bb_upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals