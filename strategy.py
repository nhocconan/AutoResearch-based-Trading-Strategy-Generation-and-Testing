#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation (2.0x 20-period average)
# Camarilla pivot levels provide intraday support/resistance. Breakouts above R3 or below S3 with 4h EMA50 trend alignment
# capture strong momentum moves. Volume confirmation filters false breakouts. Session filter (08-20 UTC) reduces noise.
# Discrete sizing 0.20 targets ~60-150 trades over 4 years (15-37/year) for 1h timeframe.

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_Trend_Volume_Session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA50 trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate EMA(50) on 4h
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1h typical price for Camarilla levels
    typical_price = (high + low + close) / 3.0
    
    # Calculate previous day's high, low, close for Camarilla levels
    # Use 24-period lookback for 1h data (24h = 1 day)
    lookback = 24
    prev_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().shift(lookback).values
    prev_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().shift(lookback).values
    prev_close = pd.Series(close).rolling(window=lookback, min_periods=lookback).mean().shift(lookback).values
    
    # Calculate Camarilla levels
    range_val = prev_high - prev_low
    camarilla_r3 = prev_close + (range_val * 1.1 / 4.0)
    camarilla_s3 = prev_close - (range_val * 1.1 / 4.0)
    
    # Volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Camarilla and EMA calculations)
    start_idx = lookback + 50  # 24 + 50 = 74
    
    for i in range(start_idx, n):
        # Check session filter
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Check for NaN values in indicators
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long breakout: price > Camarilla R3 with 4h uptrend (close > EMA50)
            long_breakout = close[i] > camarilla_r3[i]
            # Short breakdown: price < Camarilla S3 with 4h downtrend (close < EMA50)
            short_breakout = close[i] < camarilla_s3[i]
            
            # 4h EMA50 trend filter: close above/below EMA indicates trend direction
            ema_trend_up = close[i] > ema_50_4h_aligned[i]
            ema_trend_down = close[i] < ema_50_4h_aligned[i]
            
            if long_breakout and ema_trend_up and volume_spike[i]:
                signals[i] = 0.20
                position = 1
            elif short_breakout and ema_trend_down and volume_spike[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price < Camarilla S3 or trend reversal (close < EMA50)
            if close[i] < camarilla_s3[i] or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price > Camarilla R3 or trend reversal (close > EMA50)
            if close[i] > camarilla_r3[i] or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals