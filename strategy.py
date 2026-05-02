#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume spike confirmation
# Uses 1h primary timeframe for balance of signal frequency and noise reduction
# Camarilla R1/S1 levels provide high-probability breakout zones from 4h calculation
# 4h EMA50 ensures alignment with higher timeframe trend to avoid counter-trend entries
# Volume spike (>2.0 * 24-period EMA) confirms institutional participation
# Session filter (08-20 UTC) reduces noise trades outside active market hours
# Designed for low trade frequency: ~15-30 trades/year per symbol with 0.20 sizing
# Works in bull markets via breakout continuation and bear markets via trend-following alignment
# Avoids overtrading by requiring confluence of price level, trend, volume, and session

name = "1h_Camarilla_R1S1_4hEMA50_Trend_Volume_Session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Session filter: 08-20 UTC (active market hours)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h HTF data for Camarilla levels and EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h Camarilla levels (based on previous 4h bar)
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Previous 4h bar OHLC for Camarilla calculation
    prev_close_4h = np.roll(close_4h, 1)
    prev_high_4h = np.roll(high_4h, 1)
    prev_low_4h = np.roll(low_4h, 1)
    prev_close_4h[0] = np.nan
    prev_high_4h[0] = np.nan
    prev_low_4h[0] = np.nan
    
    # Calculate Camarilla R1 and S1 levels
    camarilla_r1 = np.full(len(close_4h), np.nan)
    camarilla_s1 = np.full(len(close_4h), np.nan)
    
    for i in range(1, len(close_4h)):
        hl_range = prev_high_4h[i] - prev_low_4h[i]
        camarilla_r1[i] = prev_close_4h[i] + (hl_range * 1.1 / 6)
        camarilla_s1[i] = prev_close_4h[i] - (hl_range * 1.1 / 6)
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h indicators to 1h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume confirmation: volume > 2.0 * 24-period EMA (1h * 24 = 24 periods = 1 day)
    vol_series = pd.Series(volume)
    vol_ema_24 = vol_series.ewm(span=24, adjust=False, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ema_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup: need sufficient data for all indicators
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 4h EMA50
        bullish_bias = close[i] > ema_50_4h_aligned[i]
        bearish_bias = close[i] < ema_50_4h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            if bullish_bias:
                # Long: price breaks above Camarilla R1 with volume spike
                if close[i] > camarilla_r1_aligned[i] and volume_spike[i]:
                    signals[i] = 0.20
                    position = 1
                else:
                    signals[i] = 0.0
            elif bearish_bias:
                # Short: price breaks below Camarilla S1 with volume spike
                if close[i] < camarilla_s1_aligned[i] and volume_spike[i]:
                    signals[i] = -0.20
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Avoid chop around EMA50
        
        elif position == 1:  # Long position
            # Exit: price breaks below Camarilla S1 or price below 4h EMA50
            if close[i] < camarilla_s1_aligned[i] or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price breaks above Camarilla R1 or price above 4h EMA50
            if close[i] > camarilla_r1_aligned[i] or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals