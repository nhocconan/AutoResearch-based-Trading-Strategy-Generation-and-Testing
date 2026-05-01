#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume spike confirmation.
# Camarilla pivot levels calculated from previous 4h bar: R3 = close + 1.1*(high-low)*1.1/4, S3 = close - 1.1*(high-low)*1.1/4
# Breakout above R3 with volume confirmation = long, breakdown below S3 with volume confirmation = short.
# Trend filter: price > 4h EMA50 for longs, price < 4h EMA50 for shorts.
# Uses discrete sizing 0.20 to minimize fee churn and control drawdown.
# Session filter 08-20 UTC to reduce noise. Designed for low-frequency, high-conviction trades in both bull and bear markets.

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
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 4h data ONCE before loop for HTF indicators
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Previous 4h bar's high/low/close for Camarilla pivot calculation
    prev_high_4h = df_4h['high'].values
    prev_low_4h = df_4h['low'].values
    prev_close_4h = df_4h['close'].values
    
    # Calculate Camarilla R3 and S3 levels from previous 4h bar
    # R3 = close + 1.1*(high-low)*1.1/4, S3 = close - 1.1*(high-low)*1.1/4
    camarilla_r3_4h = prev_close_4h + 1.1 * (prev_high_4h - prev_low_4h) * 1.1 / 4
    camarilla_s3_4h = prev_close_4h - 1.1 * (prev_high_4h - prev_low_4h) * 1.1 / 4
    
    # Align Camarilla levels to 1h timeframe (wait for 4h bar to close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3_4h)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3_4h)
    
    # Volume confirmation: current 1h volume > 2.0x 20-bar 1h volume average
    vol_ma_1h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma_1h * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA50
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC (reduce noise, focus on active sessions)
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            # Outside session: flatten position if any
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_ema_50_4h = ema_50_4h_aligned[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        vol_confirm = volume_confirm[i]
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: break above R3 with volume confirmation AND price > 4h EMA50 (uptrend)
            if (curr_close > curr_r3 and 
                vol_confirm and 
                curr_close > curr_ema_50_4h):
                signals[i] = 0.20
                position = 1
            # Short: break below S3 with volume confirmation AND price < 4h EMA50 (downtrend)
            elif (curr_close < curr_s3 and 
                  vol_confirm and 
                  curr_close < curr_ema_50_4h):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price < 4h EMA50 (trend violation) OR close below R3 (failed breakout)
            if (curr_close < curr_ema_50_4h or 
                curr_close < curr_r3):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price > 4h EMA50 (trend violation) OR close above S3 (failed breakout)
            if (curr_close > curr_ema_50_4h or 
                curr_close > curr_s3):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals