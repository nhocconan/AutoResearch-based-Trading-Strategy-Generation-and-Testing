#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume confirmation
# Uses 1h primary timeframe targeting 15-37 trades/year (60-150 total over 4 years)
# 4h EMA50 ensures alignment with intermediate trend to avoid counter-trend entries
# Camarilla R1/S1 levels from 4h provide intraday pivot-based breakout levels
# Volume spike (>2.0 * 20-period EMA on 1h) confirms strong participation
# Session filter (08-20 UTC) reduces noise outside active trading hours
# Discrete position sizing (0.20) minimizes fee churn while maintaining adequate exposure
# Works in bull (continuation) and bear (mean reversion via short) markets

name = "1h_Camarilla_R1S1_4hEMA50_Trend_Volume_Session"
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
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h data for EMA50 trend filter and Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h EMA50 to 1h timeframe
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla levels from previous 4h bar (HLC of completed 4h bar)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h_shifted = np.roll(close_4h, 1)  # Previous 4h bar's close
    close_4h_shifted[0] = close_4h[0]  # First bar uses its own close
    
    # Camarilla R1, S1 levels: based on previous 4h bar's range
    # R1 = close + 1.1 * (high - low) / 12
    # S1 = close - 1.1 * (high - low) / 12
    camarilla_r1 = close_4h_shifted + 1.1 * (high_4h - low_4h) / 12
    camarilla_s1 = close_4h_shifted - 1.1 * (high_4h - low_4h) / 12
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    # Volume confirmation: volume > 2.0 * 20-period EMA (1h)
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup: need sufficient data for all indicators
    start_idx = 100
    
    for i in range(start_idx, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i])):
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
                signals[i] = 0.0  # Avoid chop around 4h EMA50
        
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