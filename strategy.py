#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h EMA(21) pullback strategy with 4h trend filter and volume confirmation
# Long when: price pulls back to EMA(21) in uptrend (4h close > 4h EMA(50)) with volume > 1.5x average
# Short when: price pulls back to EMA(21) in downtrend (4h close < 4h EMA(50)) with volume > 1.5x average
# Uses 4h for trend direction (reduces whipsaw) and 1h for precise entry timing
# Session filter: 08-20 UTC to avoid low-liquidity periods
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe
# Signal size: 0.20 (discrete to minimize fee churn)

name = "1h_EMA21_Pullback_4hTrend_VolumeFilter"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Calculate volume confirmation on 1h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 4h data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50 trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1h EMA21 for pullback entries
    if len(close) >= 21:
        ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    else:
        ema_21 = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(ema_21[i]) or 
            np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price at or slightly below EMA21, uptrend on 4h, volume confirmation
            if (close[i] <= ema_21[i] * 1.001 and  # Allow small tolerance for pullback
                low[i] >= ema_21[i] * 0.995 and   # Price didn't break below EMA21 significantly
                close_4h[i//16] > ema_50_4h[i//16] if i//16 < len(ema_50_4h) else False and  # 4h uptrend
                volume_filter[i]):
                signals[i] = 0.20
                position = 1
            # Short: price at or slightly above EMA21, downtrend on 4h, volume confirmation
            elif (close[i] >= ema_21[i] * 0.999 and  # Allow small tolerance for pullback
                  high[i] <= ema_21[i] * 1.005 and   # Price didn't break above EMA21 significantly
                  close_4h[i//16] < ema_50_4h[i//16] if i//16 < len(ema_50_4h) else False and  # 4h downtrend
                  volume_filter[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price breaks above EMA21 (momentum continuation) or stops loss implicitly
            if close[i] > ema_21[i] * 1.01:  # Clear break above EMA21
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price breaks below EMA21 (momentum continuation)
            if close[i] < ema_21[i] * 0.99:  # Clear break below EMA21
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals