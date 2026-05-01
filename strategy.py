#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation
# Uses 4h EMA50 for structural trend bias (long when price > EMA50, short when price < EMA50)
# Camarilla R3/S3 breakout provides entry timing in direction of 4h trend
# Volume confirmation > 1.8x 20-period EMA ensures institutional participation
# Session filter (08-20 UTC) reduces noise trades
# Designed for low trade frequency: ~15-37 trades/year per symbol with 0.20 sizing
# 4h EMA50 filter reduces false breakouts in choppy markets while capturing strong trends
# Works in both bull and bear markets by following the dominant 4h trend

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_Trend_Volume_v1"
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
    
    # 4h HTF data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla pivot levels from previous 1h bar
    # Typical price = (H + L + C) / 3
    typical_price = (high + low + close) / 3.0
    # Use previous bar's typical price for pivot calculation (no look-ahead)
    typical_price_prev = np.roll(typical_price, 1)
    typical_price_prev[0] = np.nan
    
    # Camarilla levels based on previous 1h bar's range
    # R3 = C + (H-L) * 1.1/2
    # S3 = C - (H-L) * 1.1/2
    # where C, H, L are from previous 1h bar
    high_prev = np.roll(high, 1)
    low_prev = np.roll(low, 1)
    close_prev = np.roll(close, 1)
    high_prev[0] = np.nan
    low_prev[0] = np.nan
    close_prev[0] = np.nan
    
    camarilla_range = (high_prev - low_prev) * 1.1 / 2.0
    camarilla_R3 = close_prev + camarilla_range
    camarilla_S3 = close_prev - camarilla_range
    
    # Volume confirmation: volume > 1.8 * 20-period EMA
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ema_20)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup: need 4h data for EMA50 (50 bars) + Camarilla needs prev bar + volume EMA20
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_R3[i]) or 
            np.isnan(camarilla_S3[i]) or np.isnan(vol_ema_20[i])):
            signals[i] = 0.0
            continue
        
        # Apply session filter
        if not session_filter[i]:
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 4h EMA50: long above EMA50, short below EMA50
        bullish_bias = close[i] > ema_50_4h_aligned[i]
        bearish_bias = close[i] < ema_50_4h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            if bullish_bias:
                # Long: Camarilla R3 breakout above with volume spike
                if close[i] > camarilla_R3[i] and volume_spike[i]:
                    signals[i] = 0.20
                    position = 1
                else:
                    signals[i] = 0.0
            elif bearish_bias:
                # Short: Camarilla S3 breakdown below with volume spike
                if close[i] < camarilla_S3[i] and volume_spike[i]:
                    signals[i] = -0.20
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Avoid chop around EMA50
        
        elif position == 1:  # Long position
            # Exit: Camarilla S3 breakdown below (failure of breakout)
            if close[i] < camarilla_S3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: Camarilla R3 breakout above (failure of breakdown)
            if close[i] > camarilla_R3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals