#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1d trend filter and weekly volume confirmation
# - 1d EMA(50) defines trend direction (long when price > EMA50, short when price < EMA50)
# - Weekly volume > 1.5x 10-period average for conviction
# - 12h RSI(14) for entry timing: long when RSI < 30 in uptrend, short when RSI > 70 in downtrend
# - Exit on opposite RSI extreme (RSI > 70 for long, RSI < 30 for short) or trend reversal
# - Position size: 0.25 (25%) to balance return and drawdown
# - Designed to work in both bull and bear markets by following higher timeframe trend
# - Target: 15-30 trades/year to avoid excessive fee drag

name = "12h_EMA50_RSI_WeeklyVolume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA(50) for trend direction
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get weekly data for volume confirmation
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly volume average (10-period)
    vol_1w = df_1w['volume'].values
    vol_ma_1w = pd.Series(vol_1w).rolling(window=10, min_periods=10).mean().values
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    # 12h RSI(14) for entry timing
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_1w_aligned[i]) or np.isnan(rsi_values[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current weekly volume > 1.5x average
        volume_filter = vol_ma_1w_aligned[i] > 0 and volume[i] > 1.5 * vol_ma_1w_aligned[i]
        
        if position == 0:
            # Look for long entry: uptrend (price > 1d EMA50) + oversold RSI + volume
            if close[i] > ema_50_1d_aligned[i] and rsi_values[i] < 30 and volume_filter:
                signals[i] = 0.25
                position = 1
            # Look for short entry: downtrend (price < 1d EMA50) + overbought RSI + volume
            elif close[i] < ema_50_1d_aligned[i] and rsi_values[i] > 70 and volume_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit on overbought RSI or trend reversal
            if rsi_values[i] > 70 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit on oversold RSI or trend reversal
            if rsi_values[i] < 30 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals