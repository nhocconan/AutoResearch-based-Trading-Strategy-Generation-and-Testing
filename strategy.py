# 6h_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: Camarilla pivot levels (R3/S3) act as strong support/resistance on 6-hour timeframe.
# Breakout above R3 with daily uptrend and volume confirmation indicates bullish continuation.
# Breakdown below S3 with daily downtrend and volume confirmation indicates bearish continuation.
# Works in both bull and bear markets because trend direction is determined by daily trend, not price level.
# Uses 1d trend filter to avoid counter-trend trades, volume to confirm institutional participation.
# Target: 20-50 trades/year per symbol (80-200 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period"""
    typical_price = (high + low + close) / 3.0
    range_val = high - low
    
    # Camarilla levels
    R4 = close + range_val * 1.1 / 2
    R3 = close + range_val * 1.1 / 4
    R2 = close + range_val * 1.1 / 6
    R1 = close + range_val * 1.1 / 12
    PP = typical_price
    S1 = close - range_val * 1.1 / 12
    S2 = close - range_val * 1.1 / 6
    S3 = close - range_val * 1.1 / 4
    S4 = close - range_val * 1.1 / 2
    
    return R3, S3  # Return only the levels we need

def calculate_ema(values, period):
    """Exponential Moving Average with proper warmup"""
    if len(values) < period:
        return np.full_like(values, np.nan, dtype=np.float64)
    
    ema = np.full_like(values, np.nan, dtype=np.float64)
    alpha = 2.0 / (period + 1)
    ema[period-1] = np.mean(values[:period])
    
    for i in range(period, len(values)):
        ema[i] = alpha * values[i] + (1 - alpha) * ema[i-1]
    
    return ema

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels (R3, S3)
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    r3_vals = np.full_like(daily_close, np.nan)
    s3_vals = np.full_like(daily_close, np.nan)
    
    for i in range(len(daily_close)):
        r3, s3 = calculate_camarilla(daily_high[i], daily_low[i], daily_close[i])
        r3_vals[i] = r3
        s3_vals[i] = s3
    
    # Align Camarilla levels to 6h timeframe
    r3_6h = align_htf_to_ltf(prices, df_1d, r3_vals)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3_vals)
    
    # Calculate daily EMA34 for trend filter
    ema_34_1d = calculate_ema(daily_close, 34)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get daily volume for confirmation
    daily_volume = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(daily_volume).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need EMA (34) and volume MA (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 6h price and volume
        price_now = close[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        
        # Current levels
        r3 = r3_6h[i]
        s3 = s3_6h[i]
        ema_trend = ema_34_1d_aligned[i]
        
        # Volume filter: volume > 1.5x daily average
        vol_filter = vol_now > 1.5 * vol_ma
        
        if position == 0:
            # Breakout above R3 with daily uptrend: long
            if price_now > r3 and ema_trend > close[i-1] and vol_filter:
                signals[i] = size
                position = 1
            # Breakdown below S3 with daily downtrend: short
            elif price_now < s3 and ema_trend < close[i-1] and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to R3 or trend reverses
            if price_now < r3 or ema_trend < close[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to S3 or trend reverses
            if price_now > s3 or ema_trend > close[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0