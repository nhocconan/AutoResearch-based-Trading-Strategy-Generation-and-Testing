#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume spike confirmation
# Camarilla pivot points identify key support/resistance levels (R3/S3) for breakout trading
# 4h EMA50 ensures trades align with intermediate timeframe trend (avoid counter-trend entries)
# Volume spike (1.8x 20-period average) confirms breakout momentum
# Session filter (08-20 UTC) reduces noise trades
# Discrete sizing 0.20 minimizes fee churn. Target: 60-150 total trades over 4 years (15-37/year).

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_VolumeSpike_v1"
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
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 4h EMA50 for trend filter (loaded ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla pivot points for previous day (using daily data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Camarilla levels: based on previous day's high, low, close
    # R4 = close + ((high-low)*1.1/2)
    # R3 = close + ((high-low)*1.1/4)
    # S3 = close - ((high-low)*1.1/4)
    # S4 = close - ((high-low)*1.1/2)
    prev_close = df_1d['close'].values[:-1]  # yesterday's close
    prev_high = df_1d['high'].values[:-1]    # yesterday's high
    prev_low = df_1d['low'].values[:-1]      # yesterday's low
    
    camarilla_range = (prev_high - prev_low) * 1.1
    r3 = prev_close + camarilla_range / 4
    s3 = prev_close - camarilla_range / 4
    
    # Align Camarilla levels to 1h timeframe (each level applies to next day's 1h bars)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3, additional_delay_bars=1)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3, additional_delay_bars=1)
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 1)  # warmup for 4h EMA50 and Camarilla (need at least 1 day of data)
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_50 = ema_50_4h_aligned[i]
        curr_r3 = r3_aligned[i]
        curr_s3 = s3_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if curr_volume_spike:
                # Bullish breakout: price breaks above R3 AND above 4h EMA50 (uptrend)
                if curr_close > curr_r3 and curr_close > curr_ema_50:
                    signals[i] = 0.20
                    position = 1
                # Bearish breakout: price breaks below S3 AND below 4h EMA50 (downtrend)
                elif curr_close < curr_s3 and curr_close < curr_ema_50:
                    signals[i] = -0.20
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when price falls below 4h EMA50 (trend reversal) or below S3 (support break)
            if curr_close < curr_ema_50 or curr_close < curr_s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit when price rises above 4h EMA50 (trend reversal) or above R3 (resistance break)
            if curr_close > curr_ema_50 or curr_close > curr_r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals