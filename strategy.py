#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1w EMA50 trend filter and volume confirmation
# Uses Camarilla R3/S3 levels from daily pivots for high-probability breakout entries.
# 1w EMA50 ensures trades only with long-term trend, reducing false breakouts in choppy markets.
# Volume confirmation at 1.8x average filters low-participation moves.
# Session filter (08-20 UTC) avoids low-liquidity periods.
# Discrete sizing 0.25 to minimize fee churn. Target: 75-150 total trades over 4 years (19-37/year).
# Camarilla levels provide mathematical support/resistance that adapts to volatility, working in both bull and bear markets.

name = "12h_Camarilla_R3_S3_Breakout_1wEMA50_Volume"
timeframe = "12h"
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
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 1d OHLC for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get daily high, low, close for pivot calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels: R3, R2, R1, PP, S1, S2, S3
    # PP = (H + L + C) / 3
    # R3 = PP + (H - L) * 1.1/2
    # S3 = PP - (H - L) * 1.1/2
    pivot = (high_1d + low_1d + close_1d) / 3.0
    camarilla_r3 = pivot + (high_1d - low_1d) * 1.1 / 2.0
    camarilla_s3 = pivot - (high_1d - low_1d) * 1.1 / 2.0
    
    # Align Camarilla levels to 12h timeframe (already completed 1d bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: 1.8x 30-period average (balanced threshold)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Check for NaN values in indicators
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above Camarilla R3 AND price > 1w EMA50 AND volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Camarilla S3 AND price < 1w EMA50 AND volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price drops below Camarilla PP (pivot point) OR below 1w EMA50
            # Calculate PP for exit condition
            df_1d_current = get_htf_data(prices, '1d')
            if len(df_1d_current) >= 1:
                high_1d_curr = df_1d_current['high'].values[-1] if len(df_1d_current['high'].values) > 0 else 0
                low_1d_curr = df_1d_current['low'].values[-1] if len(df_1d_current['low'].values) > 0 else 0
                close_1d_curr = df_1d_current['close'].values[-1] if len(df_1d_current['close'].values) > 0 else 0
                pivot_curr = (high_1d_curr + low_1d_curr + close_1d_curr) / 3.0
                # Align current pivot (use previous day's pivot for current bar)
                if len(df_1d_current) >= 2:
                    high_1d_prev = df_1d_current['high'].values[-2]
                    low_1d_prev = df_1d_current['low'].values[-2]
                    close_1d_prev = df_1d_current['close'].values[-2]
                    pivot_prev = (high_1d_prev + low_1d_prev + close_1d_prev) / 3.0
                    pivot_prev_aligned = align_htf_to_ltf(prices, df_1d_current, np.full_like(df_1d_current['close'].values, pivot_prev))[i]
                else:
                    pivot_prev_aligned = pivot_curr  # fallback
            else:
                pivot_prev_aligned = camarilla_s3_aligned[i] + (camarilla_r3_aligned[i] - camarilla_s3_aligned[i]) / 2.0  # approximate
            
            if close[i] < pivot_prev_aligned or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price rises above Camarilla PP OR above 1w EMA50
            if close[i] > pivot_prev_aligned or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals