#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation.
# Long when price breaks above R3 + volume spike in bull trend (price > 4h EMA50).
# Short when price breaks below S3 + volume spike in bear trend (price < 4h EMA50).
# Uses 4h EMA50 for regime filter and 1d for higher-timeframe trend confirmation.
# Designed for 60-150 total trades over 4 years = 15-37/year on 1h timeframe.
# Works in both bull (breakouts with trend) and bear (breakdowns with trend) markets.

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_Trend_Volume"
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
    
    # Get 4h data for EMA50 trend filter and Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for higher-timeframe trend confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for higher-timeframe trend
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels from previous 4h bar
    # Typical Price = (H + L + C) / 3
    typical_price = (df_4h['high'] + df_4h['low'] + df_4h['close']) / 3
    # Range = H - L
    rang = df_4h['high'] - df_4h['low']
    # Camarilla R3 = C + (H-L) * 1.1/4
    # Camarilla S3 = C - (H-L) * 1.1/4
    r3 = typical_price + (rang * 1.1 / 4)
    s3 = typical_price - (rang * 1.1 / 4)
    
    # Align Camarilla levels to 1h timeframe (use previous 4h bar's levels)
    r3_aligned = align_htf_to_ltf(prices, df_4h, r3.values)
    s3_aligned = align_htf_to_ltf(prices, df_4h, s3.values)
    
    # Volume regime: current 1h volume > 1.5x 24-period MA (6h equivalent)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (1.5 * vol_ma_24)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Get current values
        close_val = close[i]
        ema_trend_4h = ema_50_4h_aligned[i]
        ema_trend_1d = ema_50_1d_aligned[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        vol_spike = volume_spike[i]
        
        # Skip if any value is NaN
        if np.isnan(ema_trend_4h) or np.isnan(ema_trend_1d) or np.isnan(r3_val) or np.isnan(s3_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Determine trend regime: bull if price > both 4h and 1d EMA50
        is_bull_trend = (close_val > ema_trend_4h) and (close_val > ema_trend_1d)
        # Determine trend regime: bear if price < both 4h and 1d EMA50
        is_bear_trend = (close_val < ema_trend_4h) and (close_val < ema_trend_1d)
        
        # Generate signals
        if position == 0:
            # Long entry: price breaks above R3 + volume spike + bull trend
            long_entry = (close_val > r3_val) and vol_spike and is_bull_trend
            # Short entry: price breaks below S3 + volume spike + bear trend
            short_entry = (close_val < s3_val) and vol_spike and is_bear_trend
            
            if long_entry:
                signals[i] = 0.20
                position = 1
            elif short_entry:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price breaks below S3 or trend turns bearish
            if (close_val < s3_val) or not is_bull_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price breaks above R3 or trend turns bullish
            if (close_val > r3_val) or not is_bear_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals