#!/usr/bin/env python3
"""
Hypothesis: 1-hour RSI(2) mean reversion with 4-hour trend filter and volume confirmation.
Goes long when RSI(2) < 10 and price above 4-hour EMA(50) with volume > 1.5x average.
Goes short when RSI(2) > 90 and price below 4-hour EMA(50) with volume > 1.5x average.
Uses 4-hour trend to avoid counter-trend trades in strong moves, focusing on pullbacks.
Target: 15-35 trades/year (~60-140 total over 4 years) to minimize fee drag.
Works in bull/bear by trading mean reversion within the trend.
"""
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
    
    # Get 4-hour data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4-hour EMA(50) for trend
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate RSI(2) on 1-hour data
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    avg_loss = loss.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.fillna(50).values  # Fill NaN with 50 (neutral)
    
    # Calculate 1-hour volume MA(20)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.20   # 20% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need RSI(2) and volume MA
    start_idx = max(2, 20)  # RSI(2) needs 2, vol MA needs 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        ema_50 = ema_50_4h_aligned[i]
        rsi_val = rsi[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume filter: volume > 1.5x 1-hour average
        vol_filter = vol_now > 1.5 * vol_ma
        
        # Entry conditions: RSI(2) extreme + volume + 4h trend alignment
        if position == 0:
            # Long: RSI < 10 (oversold) + volume + price above 4h EMA (uptrend)
            if rsi_val < 10 and vol_filter and close[i] > ema_50:
                signals[i] = size
                position = 1
            # Short: RSI > 90 (overbought) + volume + price below 4h EMA (downtrend)
            elif rsi_val > 90 and vol_filter and close[i] < ema_50:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI > 50 (mean reversion complete) or trend breakdown
            if rsi_val > 50 or close[i] < ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: RSI < 50 (mean reversion complete) or trend reversal
            if rsi_val < 50 or close[i] > ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_RSI2_MeanReversion_4hTrend_VolumeFilter"
timeframe = "1h"
leverage = 1.0