#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum with 4h trend filter and volume confirmation
# Uses 4h EMA50 for trend direction, 1h RSI(14) for momentum, and volume spike for confirmation
# Designed to work in both bull and bear markets by following higher timeframe trend
# Target: 15-37 trades/year (60-150 total over 4 years) with session filter (08-20 UTC)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1h RSI(14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 1h volume spike (volume > 1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    start_idx = max(50, 14, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade between 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 4h EMA50
        uptrend = close[i] > ema_50_4h_aligned[i]
        downtrend = close[i] < ema_50_4h_aligned[i]
        
        # Momentum and volume confirmation
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        vol_confirmed = volume_spike[i]
        
        if position == 0:
            # Long: uptrend + RSI oversold + volume spike
            if uptrend and rsi_oversold and vol_confirmed:
                signals[i] = 0.20
                position = 1
            # Short: downtrend + RSI overbought + volume spike
            elif downtrend and rsi_overbought and vol_confirmed:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: RSI overbought or trend breakdown
            if rsi[i] > 70 or close[i] < ema_50_4h_aligned[i]:
                signals[i] = -0.20  # reverse to short
                position = -1
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: RSI oversold or trend reversal
            if rsi[i] < 30 or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.20  # reverse to long
                position = 1
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4hEMA50_RSI_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0