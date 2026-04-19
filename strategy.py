#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h 4h/1d EMA crossover with volume and session filter.
# Use 4h EMA(21) and 1d EMA(50) for trend direction: long when 4h EMA > 1d EMA, short when opposite.
# Entry on 1h: price crosses above/below 1h EMA(13) with volume > 1.5x 20-period average.
# Session filter: only trade 08-20 UTC to avoid low-volume Asian session.
# Target: 15-30 trades/year to stay within frequency limits.
name = "1h_EMA_Crossover_Volume_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h EMA(21) for intermediate trend
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Get 1d EMA(50) for long-term trend
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 1h EMA(13) for entry trigger
    ema_1h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # 1h volume MA(20) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 21, 13, 20)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available or outside session
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(ema_1h[i]) or np.isnan(vol_ma_20[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_4h_val = ema_4h_aligned[i]
        ema_1d_val = ema_1d_aligned[i]
        ema_1h_val = ema_1h[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        # Trend determination: 4h EMA vs 1d EMA
        uptrend = ema_4h_val > ema_1d_val
        downtrend = ema_4h_val < ema_1d_val
        
        if position == 0:
            # Look for entries aligned with higher timeframe trend
            if uptrend and volume_confirmed:
                # Long when price crosses above 1h EMA in uptrend
                if price > ema_1h_val and close[i-1] <= ema_1h[i-1]:
                    signals[i] = 0.20
                    position = 1
            elif downtrend and volume_confirmed:
                # Short when price crosses below 1h EMA in downtrend
                if price < ema_1h_val and close[i-1] >= ema_1h[i-1]:
                    signals[i] = -0.20
                    position = -1
        
        elif position == 1:
            # Long exit: price crosses below 1h EMA or trend changes
            if price < ema_1h_val or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price crosses above 1h EMA or trend changes
            if price > ema_1h_val or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals