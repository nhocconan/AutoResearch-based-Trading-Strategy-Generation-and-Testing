#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot R3/S3 breakout with 4h EMA50 trend filter and volume spike confirmation
# Using 4h/1d for signal direction (trend and structure), 1h only for precise entry timing.
# Session filter (08-20 UTC) reduces noise trades during low-volume periods.
# Discrete position sizing 0.20 targets ~60-120 trades over 4 years (15-30/year) to minimize fee drag.
# Camarilla levels provide high-probability intraday S/R; breakouts with trend alignment capture momentum.
# Volume spike (2.0x 20-period average) filters false breakouts in choppy markets.

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_Trend_Volume_Session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Precompute session filter (08-20 UTC) ONCE before loop
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla pivot and EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels (R3, S3) from previous 4h bar
    typical_price = (df_4h['high'] + df_4h['low'] + df_4h['close']) / 3
    pivot = typical_price.values
    range_ = df_4h['high'].values - df_4h['low'].values
    
    # Camarilla levels: R3 = pivot + range * 1.1/4, S3 = pivot - range * 1.1/4
    camarilla_r3 = pivot + range_ * 1.1 / 4
    camarilla_s3 = pivot - range_ * 1.1 / 4
    
    # Align Camarilla levels to 1h (wait for completed 4h bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    
    # Calculate EMA(50) on 4h for trend filter
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume confirmation (2.0x 20-period average) on 1h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for EMA calculations)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Check for NaN values in indicators
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long breakout: price > R3 with 4h uptrend (close > EMA50)
            long_breakout = close[i] > camarilla_r3_aligned[i]
            # Short breakdown: price < S3 with 4h downtrend (close < EMA50)
            short_breakout = close[i] < camarilla_s3_aligned[i]
            
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
            # Exit: price < S3 or trend reversal (close < EMA50)
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price > R3 or trend reversal (close > EMA50)
            if close[i] > camarilla_r3_aligned[i] or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals