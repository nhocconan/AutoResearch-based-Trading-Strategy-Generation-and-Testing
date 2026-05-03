#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA34 trend filter and volume confirmation
# Long when price breaks above Camarilla R3 level, close > 4h EMA34, and volume > 1.8x 20-bar average
# Short when price breaks below Camarilla S3 level, close < 4h EMA34, and volume > 1.8x 20-bar average
# Uses Camarilla pivots for intraday structure, 4h EMA34 for trend filter, volume for momentum confirmation
# Designed for moderate trade frequency (~20-40/year on 1h) to minimize fee drag
# Works in bull (breakouts with rising volume) and bear (breakdowns with rising volume)
# Session filter (08-20 UTC) to avoid low-liquidity periods

name = "1h_Camarilla_R3S3_Breakout_4hEMA34_Volume_Session_v1"
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
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA34 for trend filter
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate Camarilla levels from previous 1h bar (using typical price)
    typical_price = (high + low + close) / 3.0
    camarilla_high = pd.Series(typical_price).rolling(window=20, min_periods=20).max().shift(1).values
    camarilla_low = pd.Series(typical_price).rolling(window=20, min_periods=20).min().shift(1).values
    camarilla_range = camarilla_high - camarilla_low
    camarilla_r3 = camarilla_high + 1.1 * camarilla_range / 12.0  # R3 level
    camarilla_s3 = camarilla_low - 1.1 * camarilla_range / 12.0   # S3 level
    
    # Volume confirmation (1.8x 20-period average on 1h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = max(34, 20) + 1  # EMA34(4h) + Camarilla(20) + volume MA(20) + shift(1)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_aligned[i]) or np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price > Camarilla R3, close > 4h EMA34, volume spike
            if (close[i] > camarilla_r3[i] and 
                close[i] > ema_34_aligned[i] and volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short entry: price < Camarilla S3, close < 4h EMA34, volume spike
            elif (close[i] < camarilla_s3[i] and 
                  close[i] < ema_34_aligned[i] and volume_spike[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price < Camarilla S3 or close < 4h EMA34 (trend failure)
            if (close[i] < camarilla_s3[i] or 
                close[i] < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price > Camarilla R3 or close > 4h EMA34 (trend failure)
            if (close[i] > camarilla_r3[i] or 
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals