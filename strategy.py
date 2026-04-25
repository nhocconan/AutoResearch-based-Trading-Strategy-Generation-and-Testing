#!/usr/bin/env python3
"""
6h_RSI_Divergence_1dTrend_VolumeFilter_v1
Hypothesis: Trade RSI divergences on 6h with 1d EMA50 trend filter and volume confirmation.
Long: Bullish RSI divergence (price LL, RSI HL) + price > 1d EMA50 + volume > 1.5x avg.
Short: Bearish RSI divergence (price HH, RSI LH) + price < 1d EMA50 + volume > 1.5x avg.
Exit: Opposite RSI divergence OR trend reversal.
Position size: 0.25 to manage drawdown and fees.
Target: 12-25 trades/year (50-100 total over 4 years) to stay within proven winning range for 6h.
Uses proper MTF data loading with get_htf_data() ONCE before loop and align_htf_to_ltf().
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
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for HTF trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate RSI(14) on 6h
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Volume confirmation: 6h volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for RSI (14) and volume MA (20)
    start_idx = max(14, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Check for RSI divergence (need at least 3 bars back)
        if i >= 3:
            # Bullish divergence: price makes lower low, RSI makes higher low
            bull_div = (low[i] < low[i-2]) and (rsi[i] > rsi[i-2]) and (low[i-2] < low[i-4] if i>=4 else True) and (rsi[i-2] > rsi[i-4] if i>=4 else True)
            
            # Bearish divergence: price makes higher high, RSI makes lower high
            bear_div = (high[i] > high[i-2]) and (rsi[i] < rsi[i-2]) and (high[i-2] > high[i-4] if i>=4 else True) and (rsi[i-2] < rsi[i-4] if i>=4 else True)
        else:
            bull_div = False
            bear_div = False
        
        # Determine 1d HTF trend (bullish = price above EMA50)
        htf_1d_bullish = close[i] > ema_50_1d_aligned[i]
        htf_1d_bearish = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long setup: bullish RSI divergence + 1d uptrend + volume spike
            long_setup = bull_div and htf_1d_bullish and volume_spike[i]
            
            # Short setup: bearish RSI divergence + 1d downtrend + volume spike
            short_setup = bear_div and htf_1d_bearish and volume_spike[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: bearish RSI divergence OR 1d trend turns bearish
            if bear_div or (not htf_1d_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: bullish RSI divergence OR 1d trend turns bullish
            if bull_div or (htf_1d_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_RSI_Divergence_1dTrend_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0