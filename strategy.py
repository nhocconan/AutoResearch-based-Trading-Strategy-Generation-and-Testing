#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour momentum with 4h trend filter and volume confirmation.
# Uses RSI for momentum timing, 4h EMA for trend direction, and volume surge for confirmation.
# Designed for low trade frequency (15-37/year) to minimize fee drag in 1h timeframe.
# Works in bull markets (buy pullbacks in uptrend) and bear markets (sell rallies in downtrend).
name = "1h_RSI_Pullback_4hEMA_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter (ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA (34-period) for trend direction
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate 1h RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 1h average volume (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma_20[i]
        
        # Trend filter: price above/below 4h EMA
        uptrend = close[i] > ema_34_4h_aligned[i]
        downtrend = close[i] < ema_34_4h_aligned[i]
        
        if position == 0:
            # Long: RSI < 30 (oversold) AND uptrend AND volume confirmation
            long_setup = rsi[i] < 30
            if vol_confirm and uptrend and long_setup:
                signals[i] = 0.20
                position = 1
            # Short: RSI > 70 (overbought) AND downtrend AND volume confirmation
            elif vol_confirm and downtrend and rsi[i] > 70:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: RSI > 50 (momentum fading) OR trend reversal
            exit_condition = rsi[i] > 50 or not uptrend
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: RSI < 50 (momentum fading) OR trend reversal
            exit_condition = rsi[i] < 50 or not downtrend
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals