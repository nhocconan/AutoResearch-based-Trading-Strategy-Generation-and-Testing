#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d trend following with weekly trend filter, volume confirmation, and RSI mean reversion entries
# - Weekly EMA(50) defines long-term trend direction (long when price > weekly EMA50, short when price < weekly EMA50)
# - Daily RSI(14) for entry timing: long when RSI < 30 in uptrend, short when RSI > 70 in downtrend
# - Daily volume > 1.5x 20-period average for confirmation
# - Exit on opposite RSI extreme (RSI > 70 for long, RSI < 30 for short) or trend reversal
# - Position size: 0.25 (25%) to balance return and drawdown
# - Designed to work in both bull and bear markets by following weekly trend
# - Target: 10-25 trades/year to minimize fee drag

name = "1d_EMA50_RSI_Volume_WeeklyTrend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    
    # Weekly EMA(50) for trend direction
    ema_50_weekly = pd.Series(df_weekly['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_50_weekly)
    
    # Daily volume average (20-period)
    vol_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Daily RSI(14) for entry timing
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
        if np.isnan(ema_50_weekly_aligned[i]) or np.isnan(vol_20[i]) or np.isnan(rsi_values[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > 1.5 * vol_20[i]
        
        if position == 0:
            # Look for long entry: weekly uptrend (price > weekly EMA50) + oversold RSI + volume
            if close[i] > ema_50_weekly_aligned[i] and rsi_values[i] < 30 and volume_filter:
                signals[i] = 0.25
                position = 1
            # Look for short entry: weekly downtrend (price < weekly EMA50) + overbought RSI + volume
            elif close[i] < ema_50_weekly_aligned[i] and rsi_values[i] > 70 and volume_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit on overbought RSI or trend reversal
            if rsi_values[i] > 70 or close[i] < ema_50_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit on oversold RSI or trend reversal
            if rsi_values[i] < 30 or close[i] > ema_50_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals