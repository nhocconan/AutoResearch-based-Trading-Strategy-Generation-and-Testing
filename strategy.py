#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume spike confirmation
# Targets 60-150 total trades over 4 years (15-37/year) to minimize fee drag
# Camarilla pivots provide precise intraday support/resistance levels proven effective on BTC/ETH
# 4h EMA50 determines trend bias: long when price > EMA50, short when price < EMA50
# Volume spike (2.0x 24-period average) confirms institutional participation
# Session filter (08-20 UTC) reduces noise during low-liquidity hours
# Works in bull markets via breakouts with trend alignment and bear markets via mean reversion at extremes
# Discrete position sizing: 0.20 (20% of capital) balances exposure and risk

name = "1h_Camarilla_R3S3_4hEMA50_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours once (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 4h data for EMA50 trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 (wait for completed 4h bar)
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1h Camarilla levels from prior completed 1h bar
    # Using prior 1h bar's OHLC (classic Camarilla calculation)
    if len(prices) >= 2:
        # Shifted by 1 to use prior completed bar
        prev_high = pd.Series(high).shift(1).values
        prev_low = pd.Series(low).shift(1).values
        prev_close = pd.Series(close).shift(1).values
        
        # Camarilla pivot point
        camarilla_pivot = (prev_high + prev_low + prev_close) / 3.0
        
        # Camarilla R3 and S3 levels
        camarilla_r3 = camarilla_pivot + (1.1 * (prev_high - prev_low) / 2)
        camarilla_s3 = camarilla_pivot - (1.1 * (prev_high - prev_low) / 2)
        
        # Align to 1h timeframe (already aligned by shift(1), but keep for consistency)
        camarilla_r3_aligned = camarilla_r3  # Already aligned via shift(1)
        camarilla_s3_aligned = camarilla_s3  # Already aligned via shift(1)
    else:
        return np.zeros(n)
    
    # Calculate 1h volume spike (2.0x 24-period average)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = max(24, 1)  # 24 for volume MA, 1 for shift
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Check for NaN values in indicators
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(prev_high[i]) or np.isnan(prev_low[i]) or np.isnan(prev_close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above R3 AND price > 4h EMA50 (bullish trend) AND volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_50_4h_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short entry: price breaks below S3 AND price < 4h EMA50 (bearish trend) AND volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_50_4h_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price falls below pivot point OR below 4h EMA50 (trend change)
            if close[i] < camarilla_pivot[i] or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price rises above pivot point OR above 4h EMA50 (trend change)
            if close[i] > camarilla_pivot[i] or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals