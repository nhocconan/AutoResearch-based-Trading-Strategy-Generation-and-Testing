#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h trend filter and volume confirmation
# Long when price breaks above Camarilla R3 level with 4h bullish trend (close > EMA34) and volume > 1.5x 20-period volume EMA
# Short when price breaks below Camarilla S3 level with 4h bearish trend (close < EMA34) and volume > 1.5x 20-period volume EMA
# Uses 4h EMA34 for trend filter to reduce whipsaw, targeting 15-37 trades/year on 1h.
# Volume spike filter (1.5x) is moderate to avoid overtrading. Camarilla levels provide clear structure.
# Works in bull markets via longs in bullish 4h trend regime and bear markets via shorts in bearish 4h trend regime.

name = "1h_Camarilla_R3S3_4hTrend_VolumeSpike"
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
    
    # Get 4h data for HTF trend filter - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA34 for trend filter
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_bullish_4h = close_4h > ema_34_4h
    trend_bearish_4h = close_4h < ema_34_4h
    
    # Align 4h trend to 1h timeframe
    trend_bullish_aligned = align_htf_to_ltf(prices, df_4h, trend_bullish_4h.astype(float))
    trend_bearish_aligned = align_htf_to_ltf(prices, df_4h, trend_bearish_4h.astype(float))
    
    # Calculate session filter (08-20 UTC) - precompute before loop
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Calculate Camarilla levels (R3, S3) from previous day's OHLC
    # Need daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate typical price for Camarilla: (H + L + C) / 3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    typical_price_vals = typical_price.values
    
    # Calculate Camarilla width: (H - L) * 1.1 / 8
    camarilla_width = (df_1d['high'] - df_1d['low']) * 1.1 / 8
    camarilla_width_vals = camarilla_width.values
    
    # Calculate R3 and S3 levels
    r3_levels = typical_price_vals + camarilla_width_vals * 1.1
    s3_levels = typical_price_vals - camarilla_width_vals * 1.1
    
    # Align Camarilla levels to 1h timeframe (use previous day's levels)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_levels, additional_delay_bars=1)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_levels, additional_delay_bars=1)
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)  # Volume at least 1.5x average for confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(trend_bullish_aligned[i]) or np.isnan(trend_bearish_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Apply session filter
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 AND 4h bullish trend AND volume spike
            if (close[i] > r3_aligned[i] and 
                trend_bullish_aligned[i] > 0.5 and  # 4h bullish trend
                volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: price breaks below Camarilla S3 AND 4h bearish trend AND volume spike
            elif (close[i] < s3_aligned[i] and 
                  trend_bearish_aligned[i] > 0.5 and  # 4h bearish trend
                  volume_spike[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price closes below Camarilla S3 OR 4h trend turns bearish
            if (close[i] < s3_aligned[i] or 
                trend_bearish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price closes above Camarilla R3 OR 4h trend turns bullish
            if (close[i] > r3_aligned[i] or 
                trend_bullish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals