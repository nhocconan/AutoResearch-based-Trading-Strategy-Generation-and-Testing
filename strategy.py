#!/usr/bin/env python3
"""
12h_RSI_Divergence_1wTrend_1dVolume
Hypothesis: Uses 1-week RSI divergence for momentum reversal signals, filtered by 1-week EMA trend and daily volume confirmation. 
Designed to capture medium-term reversals in both bull and bear markets by combining momentum exhaustion with trend alignment.
Targets 15-30 trades per year with low frequency to minimize fee drag on 12h timeframe.
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
    
    # Get weekly data for RSI and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Get daily data for volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Weekly RSI (14-period) for divergence detection
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Weekly EMA (34-period) for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Daily volume confirmation: current volume > 1.5 * 20-period average
    vol_avg_1d = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg_1d)
    
    # Align weekly indicators to 12h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1w, rsi)
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Start after enough data for all indicators
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(rsi_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        rsi_val = rsi_aligned[i]
        ema_trend = ema_34_aligned[i]
        vol_conf = volume_confirm[i]
        
        # Bullish divergence: price makes lower low, RSI makes higher low
        # Bearish divergence: price makes higher high, RSI makes lower high
        if i >= 2:
            price_lower_low = close[i] < close[i-1] and close[i-1] < close[i-2]
            price_higher_high = close[i] > close[i-1] and close[i-1] > close[i-2]
            rsi_higher_low = rsi_val > rsi_aligned[i-1] and rsi_aligned[i-1] > rsi_aligned[i-2]
            rsi_lower_high = rsi_val < rsi_aligned[i-1] and rsi_aligned[i-1] < rsi_aligned[i-2]
            
            bull_div = price_lower_low and rsi_higher_low
            bear_div = price_higher_high and rsi_lower_high
        else:
            bull_div = bear_div = False
        
        if position == 0:
            # Long: bullish divergence with uptrend and volume
            if bull_div and close_val > ema_trend and vol_conf:
                signals[i] = size
                position = 1
            # Short: bearish divergence with downtrend and volume
            elif bear_div and close_val < ema_trend and vol_conf:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: bearish divergence or price below EMA
            if bear_div or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: bullish divergence or price above EMA
            if bull_div or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_RSI_Divergence_1wTrend_1dVolume"
timeframe = "12h"
leverage = 1.0