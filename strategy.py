#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Volume-Weighted RSI with 1d trend filter and volatility regime
# Long when VW-RSI < 30 AND price > 1d EMA50 AND ATR ratio < 0.8 (low volatility)
# Short when VW-RSI > 70 AND price < 1d EMA50 AND ATR ratio < 0.8 (low volatility)
# Uses discrete position sizing (0.25) to minimize fee drag.
# VW-RSI gives more accurate momentum reading by weighting price changes with volume.
# Works in ranging markets via mean reversion at extremes and avoids choppy periods via volatility filter.
# Target: 15-25 trades/year on 6h timeframe (60-100 total over 4 years) to avoid overtrading.

name = "6h_VolumeWeightedRSI_1dEMA50_ATRFilter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for volatility regime filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate ATR ratio: current ATR / 50-period ATR average (regime filter)
    atr_ma_50 = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr / atr_ma_50  # < 1 = low volatility, > 1 = high volatility
    
    # Calculate Volume-Weighted RSI(14)
    # Weight price changes by volume to get more accurate momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Volume-weighted gain and loss
    vol_weighted_gain = gain * volume
    vol_weighted_loss = loss * volume
    
    # Smoothed volume-weighted gains/losses using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    avg_vol_gain = pd.Series(vol_weighted_gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_vol_loss = pd.Series(vol_weighted_loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Avoid division by zero
    rs = np.where(avg_vol_loss != 0, avg_vol_gain / avg_vol_loss, 0)
    vw_rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, 50)  # ATR and EMA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vw_rsi[i]) or 
            np.isnan(atr_ratio[i])):
            signals[i] = 0.0
            continue
        
        curr_ema50_1d = ema_50_1d_aligned[i]
        curr_vw_rsi = vw_rsi[i]
        curr_atr_ratio = atr_ratio[i]
        curr_close = close[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: VW-RSI crosses above 50 (momentum fade) OR volatility increases
            if curr_vw_rsi > 50 or curr_atr_ratio > 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: VW-RSI crosses below 50 (momentum fade) OR volatility increases
            if curr_vw_rsi < 50 or curr_atr_ratio > 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when VW-RSI < 30 AND price > 1d EMA50 AND low volatility
            if curr_vw_rsi < 30 and curr_close > curr_ema50_1d and curr_atr_ratio < 0.8:
                signals[i] = 0.25
                position = 1
            # Short when VW-RSI > 70 AND price < 1d EMA50 AND low volatility
            elif curr_vw_rsi > 70 and curr_close < curr_ema50_1d and curr_atr_ratio < 0.8:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals