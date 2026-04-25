#!/usr/bin/env python3
"""
1h_VolumeSpike_RSI_HTFTrend_v1
Hypothesis: On 1h timeframe, take mean-reversion entries when RSI(14) is extreme (<30 for long, >70 for short) only when aligned with 4h and 1d trend (price above/below EMA50) and confirmed by volume spike (vol_ratio > 2.0). Trade during UTC 08-20 session. Uses discrete sizing (0.20) and requires both HTF timeframes to agree on trend direction. Target: 15-35 trades/year by requiring tight confluence of RSI extreme, HTF trend alignment, and volume spike.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h and 1d data for HTF trend filters
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 60 or len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1d EMA50 for higher timeframe trend
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1h RSI(14) for mean reversion signals
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 1h volume ratio (current vs 24-period average)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for all indicators
    start_idx = max(60, 50, 24)
    
    for i in range(start_idx, n):
        # Skip if outside session or data not ready
        if not in_session[i] or np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Get aligned close prices for HTF trend comparison
        close_4h_aligned = align_htf_to_ltf(prices, df_4h, close_4h)[i]
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)[i]
        if np.isnan(close_4h_aligned) or np.isnan(close_1d_aligned):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
            
        # Determine 4h and 1d trend (bullish = price above EMA50)
        htf_4h_bullish = close_4h_aligned > ema_50_4h_aligned[i]
        htf_1d_bullish = close_1d_aligned > ema_50_1d_aligned[i]
        htf_bullish = htf_4h_bullish and htf_1d_bullish  # Both timeframes bullish
        htf_bearish = (not htf_4h_bullish) and (not htf_1d_bullish)  # Both timeframes bearish
        
        # Volume confirmation: need significant spike (vol_ratio > 2.0)
        volume_confirmed = vol_ratio[i] > 2.0
        
        if position == 0:
            # Long setup: RSI < 30 (oversold) + HTF bullish trend + volume confirmation
            long_setup = (rsi[i] < 30) and htf_bullish and volume_confirmed
            
            # Short setup: RSI > 70 (overbought) + HTF bearish trend + volume confirmation
            short_setup = (rsi[i] > 70) and htf_bearish and volume_confirmed
            
            if long_setup:
                signals[i] = 0.20
                position = 1
            elif short_setup:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.20
            # Exit: RSI returns to neutral (50) OR HTF trend turns bearish
            if (rsi[i] >= 50) or (not htf_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.20
            # Exit: RSI returns to neutral (50) OR HTF trend turns bullish
            if (rsi[i] <= 50) or (htf_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_VolumeSpike_RSI_HTFTrend_v1"
timeframe = "1h"
leverage = 1.0