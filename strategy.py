#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1d trend alignment (EMA50) and 1w mean reversion (RSI14).
# Enters long when price is above 1d EMA50 and 1w RSI < 30 (oversold).
# Enters short when price is below 1d EMA50 and 1w RSI > 70 (overbought).
# Uses volume confirmation (volume > 1.5x 20-period average) to filter noise.
# Designed for low-frequency, high-conviction trades targeting 12-37 trades/year.
# Works in bull markets via trend following and in bear markets via mean reversion oversold bounces.
name = "12h_1dEMA50_1wRSI_MeanReversion_Volume"
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
    open_time = prices['open_time']
    
    # Pre-compute session filter (00-23 UTC - all hours for 12h timeframe)
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = np.ones(n, dtype=bool)  # No session filter for 12h
    
    # Get 1d data for EMA50 trend (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 1w data for RSI14 mean reversion (called ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Calculate RSI(14) on weekly data
    delta = pd.Series(close_1w).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_14_1w = 100 - (100 / (1 + rs))
    rsi_14_1w_values = rsi_14_1w.values
    rsi_14_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_14_1w_values)
    
    # Volume filter: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(rsi_14_1w_aligned[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above 1d EMA50 AND 1w RSI < 30 (oversold) with volume
            if (close[i] > ema_50_1d_aligned[i] and 
                rsi_14_1w_aligned[i] < 30 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below 1d EMA50 AND 1w RSI > 70 (overbought) with volume
            elif (close[i] < ema_50_1d_aligned[i] and 
                  rsi_14_1w_aligned[i] > 70 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below 1d EMA50 or 1w RSI > 70 (overbought)
            if close[i] < ema_50_1d_aligned[i] or rsi_14_1w_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above 1d EMA50 or 1w RSI < 30 (oversold)
            if close[i] > ema_50_1d_aligned[i] or rsi_14_1w_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals