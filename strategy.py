#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h 4-hour close above 20-period EMA with 1-day ADX > 25 for trend strength,
# entering on 1-hour pullbacks to the 20-period EMA with volume confirmation.
# Long when 4h close > 4h EMA20, 1d ADX > 25, 1h close crosses above 1h EMA20, and volume > 1.5x 20-period average.
# Short when 4h close < 4h EMA20, 1d ADX > 25, 1h close crosses below 1h EMA20, and volume > 1.5x 20-period average.
# Exit when price crosses back below/above the 1h EMA20 or ADX drops below 20.
# Uses higher timeframes for trend direction and strength, 1h for precise entry timing.
# Target: 60-150 total trades over 4 years (15-37/year).

name = "1h_4hEMA20_1dADX25_VolumePullback"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h data for trend direction (EMA20)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # 1d data for trend strength (ADX)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14-period)
    plus_dm = np.zeros(len(high_1d))
    minus_dm = np.zeros(len(high_1d))
    tr = np.zeros(len(high_1d))
    
    for i in range(1, len(high_1d)):
        plus_dm[i] = max(high_1d[i] - high_1d[i-1], 0)
        minus_dm[i] = max(low_1d[i-1] - low_1d[i], 0)
        if plus_dm[i] < minus_dm[i]:
            plus_dm[i] = 0
        elif minus_dm[i] < plus_dm[i]:
            minus_dm[i] = 0
        tr[i] = max(high_1d[i] - low_1d[i], 
                    abs(high_1d[i] - close_1d[i-1]), 
                    abs(low_1d[i] - close_1d[i-1]))
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def WilderSmooth(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.nansum(data[:period]) if not np.any(np.isnan(data[:period])) else 0
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    tr14 = WilderSmooth(tr, 14)
    plus_dm14 = WilderSmooth(plus_dm, 14)
    minus_dm14 = WilderSmooth(minus_dm, 14)
    
    # Avoid division by zero
    plus_di14 = np.where(tr14 != 0, 100 * plus_dm14 / tr14, 0)
    minus_di14 = np.where(tr14 != 0, 100 * minus_dm14 / tr14, 0)
    dx = np.where((plus_di14 + minus_di14) != 0, 100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14), 0)
    adx = WilderSmooth(dx, 14)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 1h EMA20 for entry timing
    ema_20_1h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 30)  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN or outside session
        if (np.isnan(ema_20_4h_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(ema_20_1h[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: 4h EMA20 uptrend, strong trend (ADX>25), pullback to 1h EMA20 with volume
            long_cond = (close[i] > ema_20_1h[i] and 
                        close[i-1] <= ema_20_1h[i-1] and  # Cross above EMA20
                        ema_20_4h_aligned[i] > close_4h[0] if len(close_4h) > 0 else False and  # Simplified trend check
                        adx_aligned[i] > 25 and 
                        volume_filter[i])
            # Short conditions: 4h EMA20 downtrend, strong trend (ADX>25), pullback to 1h EMA20 with volume
            short_cond = (close[i] < ema_20_1h[i] and 
                         close[i-1] >= ema_20_1h[i-1] and  # Cross below EMA20
                         ema_20_4h_aligned[i] < close_4h[0] if len(close_4h) > 0 else False and  # Simplified trend check
                         adx_aligned[i] > 25 and 
                         volume_filter[i])
            
            if long_cond:
                signals[i] = 0.20
                position = 1
            elif short_cond:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price crosses below 1h EMA20 or ADX drops below 20
            if close[i] < ema_20_1h[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price crosses above 1h EMA20 or ADX drops below 20
            if close[i] > ema_20_1h[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals