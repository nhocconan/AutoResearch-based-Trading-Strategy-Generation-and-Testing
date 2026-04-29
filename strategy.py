#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation
# Uses Camarilla pivot levels for precise breakout entries with strong HTF trend alignment
# 4h EMA50 provides robust trend filter to avoid counter-trend trades in choppy markets
# Volume spike (1.8x 20-period average) confirms breakout validity with institutional participation
# Session filter (08-20 UTC) reduces noise trades during low-liquidity periods
# Fixed position size of 0.20 to control risk and minimize fee churn
# Designed for low trade frequency (target: 15-37 trades/year) to overcome 1h timeframe challenges
# Works in bull markets via upper Camarilla breaks and in bear markets via lower Camarilla breaks

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_VolumeConfirm_v1"
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
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate ATR for reference (using 14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Calculate Camarilla levels using previous day's OHLC
        # Need to get daily data for pivot calculation
        if i >= 24:  # Need at least 24 hours of data for previous day
            # Get previous day's OHLC (24 hours ago to 48 hours ago)
            prev_day_high = np.max(high[i-48:i-24])
            prev_day_low = np.min(low[i-48:i-24])
            prev_day_close = close[i-24]
            
            # Calculate Camarilla levels
            range_val = prev_day_high - prev_day_low
            camarilla_r3 = prev_day_close + range_val * 1.1 / 4
            camarilla_s3 = prev_day_close - range_val * 1.1 / 4
        else:
            camarilla_r3 = 0.0
            camarilla_s3 = 0.0
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_4h = ema_50_4h_aligned[i]
        curr_atr = atr[i]
        
        # Volume spike confirmation: current volume > 1.8x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = 0.0
        vol_spike = volume[i] > 1.8 * vol_ma_20 if vol_ma_20 > 0 else False
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: price below Camarilla S3 OR price < 4h EMA50 (trend change)
            if curr_close < camarilla_s3 or curr_close < curr_ema_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price above Camarilla R3 OR price > 4h EMA50 (trend change)
            if curr_close > camarilla_r3 or curr_close > curr_ema_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above Camarilla R3 AND price > 4h EMA50 AND volume spike
            if curr_high > camarilla_r3 and curr_close > curr_ema_4h and vol_spike:
                signals[i] = 0.20
                position = 1
            # Short entry: price breaks below Camarilla S3 AND price < 4h EMA50 AND volume spike
            elif curr_low < camarilla_s3 and curr_close < curr_ema_4h and vol_spike:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
    
    return signals