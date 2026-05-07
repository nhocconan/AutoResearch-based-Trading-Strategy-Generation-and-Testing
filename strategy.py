#!/usr/bin/env python3
"""
4H_Trend_Reversal_With_Volume_Spike
Hypothesis: On 4H timeframe, enter counter-trend positions after strong price moves (ATR-based) show exhaustion,
confirmed by volume spike and RSI extremes. Uses 1D trend filter to avoid counter-trend trades in strong trends.
Designed for both bull and bear markets: captures mean reversions during pullbacks in trending markets and 
reversals in ranging markets. Targets 20-50 trades/year to minimize fee drag.
"""
name = "4H_Trend_Reversal_With_Volume_Spike"
timeframe = "4h"
leverage = 1.0

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
    
    # Calculate ATR for move detection (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate RSI (14-period) for overbought/oversold
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Get 1D data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1D EMA50 for trend direction
    close_1d_series = pd.Series(df_1d['close'])
    ema_50 = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume filter: current 4H volume > 2.0 x 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_avg * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    start_idx = max(50, 20, 14)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(atr[i]) or np.isnan(rsi[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 40 bars between trades (~10 days on 4H TF) to reduce frequency
            if bars_since_exit < 40:
                continue
                
            # Calculate price deviation from recent mean (20-period SMA)
            sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
            price_dev = (close[i] - sma_20[i]) / atr[i] if atr[i] > 0 else 0
            
            # Long: price moved down >2 ATR, RSI oversold, volume spike, and 1D trend not strongly down
            if (price_dev < -2.0 and rsi[i] < 30 and volume_spike[i] and 
                close[i] > ema_50_aligned[i]):  # Avoid strong downtrend
                signals[i] = 0.25
                position = 1
                bars_since_exit = 0
            # Short: price moved up >2 ATR, RSI overbought, volume spike, and 1D trend not strongly up
            elif (price_dev > 2.0 and rsi[i] > 70 and volume_spike[i] and 
                  close[i] < ema_50_aligned[i]):  # Avoid strong uptrend
                signals[i] = -0.25
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit: RSI returns to neutral zone (40-60) or opposite extreme
            if position == 1 and (rsi[i] > 50 or rsi[i] > 70):
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            elif position == -1 and (rsi[i] < 50 or rsi[i] < 30):
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals