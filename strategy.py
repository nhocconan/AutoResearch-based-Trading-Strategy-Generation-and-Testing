#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h ADX Trend Strength with Volume Confirmation
# Hypothesis: Strong trends (ADX > 25) with volume confirmation provide
# reliable directional moves. Uses +DI/-DI crossovers for entries.
# ADX filters out ranging markets, reducing false signals.
# Works in both bull and bear markets by following the trend direction.
# Target: 20-40 trades/year (80-160 over 4 years).

name = "4h_adx_trend_strength_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend context
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    # Calculate ADX (+DI, -DI) on 4h data
    period = 14
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth with Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        for i in range(len(data)):
            if np.isnan(result[i-1]) if i > 0 else True:
                result[i] = data[i]
            else:
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    tr_sum = wilders_smoothing(tr, period)
    plus_dm_sum = wilders_smoothing(plus_dm, period)
    minus_dm_sum = wilders_smoothing(minus_dm, period)
    
    # Avoid division by zero
    plus_di = 100 * plus_dm_sum / tr_sum
    minus_di = 100 * minus_dm_sum / tr_sum
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, period)
    
    # Daily trend filter: price above/below 50 EMA
    close_series = pd.Series(close)
    ema_50_daily = close_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    daily_ema_aligned = align_htf_to_ltf(prices, df_daily, ema_50_daily)
    
    # Volume filter: volume > 1.3x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(adx[i]) or np.isnan(plus_di[i]) or np.isnan(minus_di[i]) or
            np.isnan(daily_ema_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: ADX weakens OR DI cross turns bearish OR trend filter fails OR volume drops
            if (adx[i] < 20 or minus_di[i] > plus_di[i] or close[i] < daily_ema_aligned[i] or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: ADX weakens OR DI cross turns bullish OR trend filter fails OR volume drops
            if (adx[i] < 20 or plus_di[i] > minus_di[i] or close[i] > daily_ema_aligned[i] or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: ADX strong AND +DI crosses above -DI AND price above daily EMA AND volume
            if (adx[i] > 25 and plus_di[i] > minus_di[i] and 
                plus_di[i-1] <= minus_di[i-1] and  # Cross just happened
                close[i] > daily_ema_aligned[i] and vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: ADX strong AND -DI crosses above +DI AND price below daily EMA AND volume
            elif (adx[i] > 25 and minus_di[i] > plus_di[i] and 
                  minus_di[i-1] <= plus_di[i-1] and  # Cross just happened
                  close[i] < daily_ema_aligned[i] and vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals