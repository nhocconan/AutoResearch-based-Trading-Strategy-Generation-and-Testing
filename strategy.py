#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation (>1.3x 20-bar MA)
# Camarilla pivot levels provide high-probability reversal/breakout zones, 4h EMA50 filters counter-trend noise, volume spike confirms participation
# Works in bull markets (breakouts continue) and bear markets (breakdowns continue) when aligned with higher timeframe trend
# Discrete sizing (0.20) minimizes fee churn. Target: 60-150 total trades over 4 years (15-37/year) with session filter (08-20 UTC) to reduce noise.

name = "1h_Camarilla_R3_S3_Breakout_4hEMA50_Trend_VolumeSpike_v1"
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
    open_time = prices['open_time'].values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h HTF data for EMA calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA(50) on 4h close
    ema_4h_50 = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h EMA to 1h timeframe
    ema_4h_50_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_50)
    
    # Previous day's high/low for Camarilla calculation (using 1d data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's high, low, close
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Align previous day's OHLC to 1h timeframe
    prev_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low)
    prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close)
    
    # Camarilla R3 and S3 levels
    # R3 = close + (high - low) * 1.1/4
    # S3 = close - (high - low) * 1.1/4
    camarilla_range = prev_high_aligned - prev_low_aligned
    r3 = prev_close_aligned + camarilla_range * 1.1 / 4
    s3 = prev_close_aligned - camarilla_range * 1.1 / 4
    
    # Volume confirmation: current volume > 1.3 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 20  # Need 20 for volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_4h_50_aligned[i]) or np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        # Camarilla breakout conditions
        upper_break = curr_close > r3[i]   # Break above R3
        lower_break = curr_close < s3[i]   # Break below S3
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above R3, above 4h EMA50, volume spike
            if upper_break and curr_close > ema_4h_50_aligned[i] and vol_spike:
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below S3, below 4h EMA50, volume spike
            elif lower_break and curr_close < ema_4h_50_aligned[i] and vol_spike:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on price below S3 or below 4h EMA50
            if curr_close < s3[i] or curr_close < ema_4h_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit on price above R3 or above 4h EMA50
            if curr_close > r3[i] or curr_close > ema_4h_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals