#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h trend filter (EMA50) and 1d volume confirmation
# Long when: price breaks above Camarilla R3 AND 4h EMA50 trending up AND 1d volume > 1.5x 20 EMA
# Short when: price breaks below Camarilla S3 AND 4h EMA50 trending down AND 1d volume > 1.5x 20 EMA
# Uses Camarilla pivots for precise intraday support/resistance, 4h EMA for trend direction,
# and 1d volume spike to confirm institutional participation. Designed for 15-37 trades/year
# with discrete sizing (0.20) to minimize fee drag. Works in bull markets via longs in uptrends
# and bear markets via shorts in downtrends, with session filter (08-20 UTC) to avoid Asian session noise.

name = "1h_Camarilla_R3S3_4hEMA50_1dVolConfirm"
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
    
    # Pre-compute session filter (08-20 UTC) - prices.index is already DatetimeIndex
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA50 trend filter - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    # Calculate 4h EMA50
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    # 4h trend: up when close > EMA50, down when close < EMA50
    trend_up_4h = close_4h > ema_50_4h
    trend_down_4h = close_4h < ema_50_4h
    # Align 4h trend to 1h timeframe
    trend_up_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_up_4h.astype(float))
    trend_down_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_down_4h.astype(float))
    
    # Get 1d data for volume confirmation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    # Calculate 1d volume EMA20
    vol_ema_20_1d = pd.Series(volume_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    # 1d volume spike: current volume > 1.5x 20 EMA
    volume_spike_1d = volume_1d > (vol_ema_20_1d * 1.5)
    # Align 1d volume spike to 1h timeframe
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    # Calculate Camarilla levels for 1h timeframe using previous bar's OHLC
    # Camarilla: R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    # We need previous bar's OHLC, so we shift by 1
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    # First bar: use current values (will be filtered out by min_periods anyway)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    camarilla_range = prev_high - prev_low
    r3 = prev_close + (camarilla_range * 1.1 / 4)
    s3 = prev_close - (camarilla_range * 1.1 / 4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):  # Start from 1 to have previous bar data
        # Skip if any value is NaN or not in session
        if (np.isnan(trend_up_4h_aligned[i]) or np.isnan(trend_down_4h_aligned[i]) or 
            np.isnan(volume_spike_1d_aligned[i]) or np.isnan(r3[i]) or np.isnan(s3[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R3 AND 4h trend up AND 1d volume spike
            if (close[i] > r3[i] and 
                trend_up_4h_aligned[i] > 0.5 and 
                volume_spike_1d_aligned[i] > 0.5):
                signals[i] = 0.20
                position = 1
            # Short conditions: price breaks below S3 AND 4h trend down AND 1d volume spike
            elif (close[i] < s3[i] and 
                  trend_down_4h_aligned[i] > 0.5 and 
                  volume_spike_1d_aligned[i] > 0.5):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price breaks below S3 OR 4h trend turns down
            if (close[i] < s3[i] or 
                trend_down_4h_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price breaks above R3 OR 4h trend turns up
            if (close[i] > r3[i] or 
                trend_up_4h_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals