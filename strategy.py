#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily timeframe with weekly trend filter and volume confirmation
# - Weekly EMA(13) defines trend direction (long when close > EMA13, short when close < EMA13)
# - Daily RSI(14) for entry timing: long when RSI < 30 in uptrend, short when RSI > 70 in downtrend
# - Volume confirmation: daily volume > 1.5x 20-period average for conviction
# - Exit on opposite RSI extreme (RSI > 70 for long, RSI < 30 for short) or trend reversal
# - Position size: 0.25 (25%) to manage drawdown
# - Designed for low frequency (~10-25 trades/year) to minimize fee drag
# - Works in both bull and bear markets by following higher timeframe trend

name = "1d_EMA13_RSI_Volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    
    # Weekly EMA(13) for trend direction
    ema_13_weekly = pd.Series(df_weekly['close'].values).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_aligned = align_htf_to_ltf(prices, df_weekly, ema_13_weekly)
    
    # Daily RSI(14) for entry timing
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Daily volume average (20-period)
    vol_ma_daily = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for RSI and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_13_aligned[i]) or np.isnan(rsi_values[i]) or np.isnan(vol_ma_daily[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current volume > 1.5x average
        volume_filter = vol_ma_daily[i] > 0 and volume[i] > 1.5 * vol_ma_daily[i]
        
        if position == 0:
            # Look for long entry: uptrend (close > weekly EMA13) + oversold RSI + volume
            if close[i] > ema_13_aligned[i] and rsi_values[i] < 30 and volume_filter:
                signals[i] = 0.25
                position = 1
            # Look for short entry: downtrend (close < weekly EMA13) + overbought RSI + volume
            elif close[i] < ema_13_aligned[i] and rsi_values[i] > 70 and volume_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit on overbought RSI or trend reversal
            if rsi_values[i] > 70 or close[i] < ema_13_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit on oversold RSI or trend reversal
            if rsi_values[i] < 30 or close[i] > ema_13_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals