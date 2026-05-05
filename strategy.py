#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume spike confirmation
# Long when price breaks above Camarilla R1 level AND price > EMA50(4h) AND volume > 2.0x 20-period average
# Short when price breaks below Camarilla S1 level AND price < EMA50(4h) AND volume > 2.0x 20-period average
# Exit when price crosses back below Camarilla S1 (for longs) or above R1 (for shorts) OR trend flips (price crosses EMA50(4h))
# Camarilla pivot levels provide intraday support/resistance structure proven effective on ETH/USDT
# 4h EMA50 provides higher timeframe trend filter to avoid counter-trend whipsaws
# Volume spike confirms institutional participation
# Target: 15-37 trades/year per symbol (60-150 total over 4 years)
# Discrete sizing (0.20) to limit fee drag
# Session filter (08-20 UTC) to reduce noise trades

name = "1h_Camarilla_R1_S1_Breakout_4hEMA50_Trend_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours for filtering (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(open_time).hour
    
    # Get 4h data ONCE before loop for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate EMA50 on 4h close for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h EMA50 to 1h timeframe
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla pivot levels from previous day's OHLC
    # Requires daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Camarilla levels: based on previous day's OHLC
    # R1 = close + 1.0833*(high-low), S1 = close - 1.0833*(high-low)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate pivot levels
    camarilla_r1 = prev_close + 1.0833 * (prev_high - prev_low)
    camarilla_s1 = prev_close - 1.0833 * (prev_high - prev_low)
    
    # Align daily Camarilla levels to 1h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume confirmation: volume > 2.0x 20-period average (spike filter)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R1 AND price > EMA50(4h) AND volume spike
            if (close[i] > camarilla_r1_aligned[i] and 
                close[i] > ema_50_4h_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: price breaks below Camarilla S1 AND price < EMA50(4h) AND volume spike
            elif (close[i] < camarilla_s1_aligned[i] and 
                  close[i] < ema_50_4h_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price crosses back below Camarilla S1 (mean reversion) OR price < EMA50(4h) (trend flip)
            if (close[i] < camarilla_s1_aligned[i] or 
                close[i] < ema_50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price crosses back above Camarilla R1 (mean reversion) OR price > EMA50(4h) (trend flip)
            if (close[i] > camarilla_r1_aligned[i] or 
                close[i] > ema_50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals