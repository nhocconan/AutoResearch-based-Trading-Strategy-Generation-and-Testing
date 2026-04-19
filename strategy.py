#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA (Kaufman Adaptive Moving Average) with 1w trend filter and volume confirmation
# - KAMA adapts to market noise: follows price closely in trending markets, stays flat in ranging markets
# - Only take long when price > KAMA AND price > 1w EMA50 (uptrend on both timeframes)
# - Only take short when price < KAMA AND price < 1w EMA50 (downtrend on both timeframes)
# - Volume confirmation: current 1d volume > 1.5x 20-period average volume
# - Designed to reduce whipsaws in ranging markets while capturing trends in both bull and bear markets
# - Target: 15-25 trades/year to minimize fee drag

name = "1d_KAMA_1wTrend_Volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # 1w EMA(50) for trend direction
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # KAMA (Kaufman Adaptive Moving Average) parameters
    er_period = 10  # Efficiency Ratio period
    fast_sc = 2/(2+1)  # smoothing constant for fastest EMA
    slow_sc = 2/(30+1)  # smoothing constant for slowest EMA
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close, n=er_period))  # absolute change over er_period periods
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # sum of absolute changes over er_period
    
    # Avoid division by zero
    er = np.zeros_like(change)
    mask = volatility != 0
    er[mask] = change[mask] / volatility[mask]
    
    # Calculate smoothing constant (SC)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[er_period] = close[er_period]  # seed value
    
    for i in range(er_period + 1, n):
        if not np.isnan(sc[i-er_period]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i-er_period] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, er_period + 20)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(kama[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current volume > 1.5x 20-period average volume
        volume_filter = vol_ma_20[i] > 0 and volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Look for long entry: uptrend on both timeframes + volume
            if close[i] > kama[i] and close[i] > ema_50_1w_aligned[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Look for short entry: downtrend on both timeframes + volume
            elif close[i] < kama[i] and close[i] < ema_50_1w_aligned[i] and volume_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when trend breaks on either timeframe
            if close[i] <= kama[i] or close[i] <= ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when trend breaks on either timeframe
            if close[i] >= kama[i] or close[i] >= ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals