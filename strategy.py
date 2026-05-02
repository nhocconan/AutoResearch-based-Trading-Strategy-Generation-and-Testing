#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume confirmation + session filter (08-20 UTC)
# Camarilla R1/S1 are tighter levels than R3/S3, capturing earlier momentum shifts
# 4h EMA50 provides trend filter to avoid counter-trend trades
# Volume confirmation (2.0x 24-period average) ensures institutional participation
# Session filter (08-20 UTC) reduces noise during low-liquidity hours
# Discrete sizing 0.20 targets 60-150 trades over 4 years (15-37/year) for 1h timeframe
# Works in bull/bear by only taking breakouts in direction of 4h EMA50 trend

name = "1h_Camarilla_R1S1_Breakout_4hEMA50_Trend_Volume_Session_v1"
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
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA50 trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 60:
        return np.zeros(n)
    
    # Calculate EMA(50) on 4h
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for Camarilla pivot levels (R1, S1)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Camarilla levels: based on previous day's range
    # R1 = close + 0.109*(high-low), S1 = close - 0.109*(high-low)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla R1 and S1
    camarilla_r1 = prev_close + 0.109 * (prev_high - prev_low)
    camarilla_s1 = prev_close - 0.109 * (prev_high - prev_low)
    
    # Align Camarilla levels to 1h timeframe (completed 1d bar only)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume confirmation (2.0x 24-period average)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for calculations)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Check for NaN values in indicators
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long breakout: price > Camarilla R1 with 4h uptrend (close > EMA50)
            long_breakout = close[i] > camarilla_r1_aligned[i]
            # Short breakdown: price < Camarilla S1 with 4h downtrend (close < EMA50)
            short_breakout = close[i] < camarilla_s1_aligned[i]
            
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
            # Exit: price < Camarilla S1 or trend reversal (close < EMA50)
            if close[i] < camarilla_s1_aligned[i] or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price > Camarilla R1 or trend reversal (close > EMA50)
            if close[i] > camarilla_r1_aligned[i] or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals