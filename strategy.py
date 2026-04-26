#!/usr/bin/env python3
"""
6h_RSI_EMA_Divergence_1dTrend_VolumeFilter
Hypothesis: Combines RSI divergence detection with EMA trend filter on 6h timeframe.
Long when: bullish RSI divergence (price makes lower low, RSI makes higher low) AND price > 6h EMA50 AND 1d uptrend AND volume spike.
Short when: bearish RSI divergence (price makes higher high, RSI makes lower high) AND price < 6h EMA50 AND 1d downtrend AND volume filter.
Uses volume confirmation to avoid false signals and discrete position sizing (0.25) to minimize fee churn.
Designed to catch reversals in both bull and bear markets by following 1d trend while using RSI divergence for precise entry timing.
Target: 12-37 trades/year (50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA34 trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    uptrend_1d = close > ema_34_1d_aligned
    downtrend_1d = close < ema_34_1d_aligned
    
    # 6h indicators
    # EMA50 for trend
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume confirmation: volume > 1.5x 20-period MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    # RSI divergence detection (lookback 5 periods)
    def detect_bullish_div(idx):
        if idx < 5:
            return False
        # Price makes lower low, RSI makes higher low
        price_lower_low = low[idx] < low[idx-5] and low[idx] == np.min(low[idx-5:idx+1])
        rsi_higher_low = rsi_values[idx] > rsi_values[idx-5] and rsi_values[idx] == np.max(rsi_values[idx-5:idx+1])
        return price_lower_low and rsi_higher_low
    
    def detect_bearish_div(idx):
        if idx < 5:
            return False
        # Price makes higher high, RSI makes lower high
        price_higher_high = high[idx] > high[idx-5] and high[idx] == np.max(high[idx-5:idx+1])
        rsi_lower_high = rsi_values[idx] < rsi_values[idx-5] and rsi_values[idx] == np.min(rsi_values[idx-5:idx+1])
        return price_higher_high and rsi_lower_high
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for EMA50 + 14 for RSI + 20 for volume MA + 5 for divergence)
    start_idx = 55
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(ema_50[i]) or 
            np.isnan(rsi_values[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: bullish RSI divergence + price > EMA50 + 1d uptrend + volume spike
            if (detect_bullish_div(i) and close[i] > ema_50[i] and 
                uptrend_1d[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: bearish RSI divergence + price < EMA50 + 1d downtrend + volume spike
            elif (detect_bearish_div(i) and close[i] < ema_50[i] and 
                  downtrend_1d[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: 1d trend changes to downtrend OR price crosses below EMA50
            if (not uptrend_1d[i] or close[i] < ema_50[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: 1d trend changes to uptrend OR price crosses above EMA50
            if (not downtrend_1d[i] or close[i] > ema_50[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_RSI_EMA_Divergence_1dTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0