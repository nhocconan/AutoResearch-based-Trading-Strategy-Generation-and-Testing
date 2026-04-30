#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h trend filter (EMA21) and volume confirmation (>1.8x 20-bar avg).
# Uses 4h EMA21 for higher timeframe trend filter to avoid counter-trend trades.
# Volume confirmation reduces false breakouts. Session filter (08-20 UTC) avoids low-liquidity periods.
# Discrete position sizing at ±0.20 to minimize fee drag while maintaining exposure.
# Target: 80-150 total trades over 4 years (20-37/year) to stay within 1h limits.
# Works in bull markets via breakout continuation and in bear markets via mean-reversion exits when price retests pivot levels.

name = "1h_Camarilla_R3S3_Breakout_4hEMA21_Trend_VolumeConfirm_Session_v1"
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
    
    # Pre-compute session hours (08-20 UTC) to avoid look-ahead
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for EMA21 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h EMA21 for trend filter
    close_4h = df_4h['close'].values
    ema_21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # warmup for EMA21
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(ema_21_4h_aligned[i]) or 
            np.isnan(volume_confirm[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_ema_21_4h = ema_21_4h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Calculate Camarilla pivot levels from previous 1h bar
            if i >= 1:
                prev_high = high[i-1]
                prev_low = low[i-1]
                prev_close = close[i-1]
                pivot_point = (prev_high + prev_low + prev_close) / 3
                camarilla_r3 = pivot_point + (prev_high - prev_low) * 1.1 / 2
                camarilla_s3 = pivot_point - (prev_high - prev_low) * 1.1 / 2
                
                # Long: price breaks above R3, close > 4h EMA21, volume spike, in session
                if (curr_close > camarilla_r3 and 
                    curr_close > curr_ema_21_4h and 
                    volume_confirm[i]):
                    signals[i] = 0.20
                    position = 1
                # Short: price breaks below S3, close < 4h EMA21, volume spike, in session
                elif (curr_close < camarilla_s3 and 
                      curr_close < curr_ema_21_4h and 
                      volume_confirm[i]):
                    signals[i] = -0.20
                    position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price moves back below R3 (mean reversion)
            if i >= 1:
                prev_high = high[i-1]
                prev_low = low[i-1]
                prev_close = close[i-1]
                pivot_point = (prev_high + prev_low + prev_close) / 3
                camarilla_r3 = pivot_point + (prev_high - prev_low) * 1.1 / 2
                
                if curr_close < camarilla_r3:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit condition: price moves back above S3 (mean reversion)
            if i >= 1:
                prev_high = high[i-1]
                prev_low = low[i-1]
                prev_close = close[i-1]
                pivot_point = (prev_high + prev_low + prev_close) / 3
                camarilla_s3 = pivot_point - (prev_high - prev_low) * 1.1 / 2
                
                if curr_close > camarilla_s3:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals