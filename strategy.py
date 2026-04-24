#!/usr/bin/env python3
"""
Hypothesis: 1h RSI(14) mean reversion with 4h EMA50 trend filter and volume confirmation.
- Long when RSI < 30 (oversold) and close > 4h EMA50 (bullish trend)
- Short when RSI > 70 (overbought) and close < 4h EMA50 (bearish trend)
- Volume must be > 1.5x 20-period average for confirmation
- Exit when RSI returns to neutral zone (40-60) or opposite extreme
- Uses 1h primary timeframe with 4h HTF to target 60-150 trades over 4 years (15-37/year)
- RSI captures mean reversion in ranging markets while trend filter avoids counter-trend traps
- Session filter (08-20 UTC) reduces noise during low-liquidity hours
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI(14): 100 - (100 / (1 + RS))
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0.0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    # Get 4h data ONCE before loop for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h EMA50 to 1h timeframe
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume confirmation: > 1.5x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * vol_ma
    
    # Session filter: 08-20 UTC (pre-compute hours array)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Start from index where all indicators are ready
    start_idx = max(14, 50, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(rsi[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            continue
        
        # Apply session filter
        if not session_filter[i]:
            signals[i] = 0.0
            continue
        
        # Long: RSI oversold (< 30), trend up (close > EMA50), volume confirmation
        if rsi[i] < 30.0 and close[i] > ema_50_4h_aligned[i] and volume_confirm[i]:
            signals[i] = 0.20
        # Short: RSI overbought (> 70), trend down (close < EMA50), volume confirmation
        elif rsi[i] > 70.0 and close[i] < ema_50_4h_aligned[i] and volume_confirm[i]:
            signals[i] = -0.20
        # Exit: RSI returns to neutral zone (40-60)
        elif 40.0 <= rsi[i] <= 60.0:
            signals[i] = 0.0
        # Hold current signal if no exit condition
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "1h_RSI_MeanReversion_4hEMA50_VolumeConfirm_SessionFilter_v1"
timeframe = "1h"
leverage = 1.0