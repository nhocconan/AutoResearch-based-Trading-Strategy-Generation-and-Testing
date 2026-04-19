#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with 1d trend filter and 6h momentum confirmation
# - 1d EMA(50) defines trend direction (long when close > EMA50, short when close < EMA50)
# - 6h RSI(14) for entry timing: long when RSI < 30 in uptrend, short when RSI > 70 in downtrend
# - 6h volume > 1.5x 20-period average for confirmation
# - Exit on opposite RSI extreme or trend reversal
# - Position size: 0.25 (25%) to balance return and drawdown
# - Designed to work in both bull and bear markets by following higher timeframe trend
# - Target: 15-30 trades/year to avoid excessive fee drain

name = "6h_EMA50_RSI_Volume_v1"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA(50) for trend direction
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 6h RSI(14) for entry timing
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # 6h volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(rsi_values[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current volume > 1.5x average
        volume_filter = vol_ma[i] > 0 and volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Look for long entry: uptrend (close > 1d EMA50) + oversold RSI + volume
            if close[i] > ema_50_1d_aligned[i] and rsi_values[i] < 30 and volume_filter:
                signals[i] = 0.25
                position = 1
            # Look for short entry: downtrend (close < 1d EMA50) + overbought RSI + volume
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