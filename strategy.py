#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume spike
# Uses 4h/1d for signal direction (trend regime and volatility filter), 1h for precise entry timing
# Camarilla levels from prior 4h bar provide intraday support/resistance; breakouts capture momentum
# 4h EMA50 ensures alignment with intermediate trend; volume >1.8x confirms participation
# Session filter (08-20 UTC) reduces noise trades during low-liquidity periods
# Discrete sizing (0.20) minimizes fee churn; target 60-150 total trades over 4 years

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_Trend_VolumeSpike_v1"
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
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate ATR for volatility (14-period)
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr2.iloc[0] = 0
    tr3.iloc[0] = 0
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume confirmation: volume > 1.8x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_confirm = volume > (1.8 * vol_ma_30)
    
    # Precompute 4h data for Camarilla levels
    df_4h_cam = get_htf_data(prices, '4h')
    if len(df_4h_cam) < 2:
        return np.zeros(n)
    
    daily_high = df_4h_cam['high'].values
    daily_low = df_4h_cam['low'].values
    daily_close = df_4h_cam['close'].values
    
    daily_high_aligned = align_htf_to_ltf(prices, df_4h_cam, daily_high)
    daily_low_aligned = align_htf_to_ltf(prices, df_4h_cam, daily_low)
    daily_close_aligned = align_htf_to_ltf(prices, df_4h_cam, daily_close)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(60, 30, 14, 50)  # warmup: need 60 1h bars (~2.5 days) for stability
    
    for i in range(start_idx, n):
        # Skip if indicators not ready or outside session
        if (np.isnan(atr[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(vol_ma_30[i]) or
            np.isnan(daily_high_aligned[i]) or
            np.isnan(daily_low_aligned[i]) or
            np.isnan(daily_close_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
            
        curr_high = high[i]
        curr_low = low[i]
        curr_close = close[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema_50_4h = ema_50_4h_aligned[i]
        
        # Use previous 4h bar's levels (shift by 1)
        prev_high = daily_high_aligned[i-1]
        prev_low = daily_low_aligned[i-1]
        prev_close = daily_close_aligned[i-1]
        
        if position == 0:  # Flat - look for new entries
            if not (np.isnan(prev_high) or np.isnan(prev_low) or np.isnan(prev_close)):
                # Calculate Camarilla levels
                range_val = prev_high - prev_low
                r3 = prev_close + (range_val * 1.1 / 4)
                s3 = prev_close - (range_val * 1.1 / 4)
                
                # Only trade with volume confirmation and trend filter
                if curr_volume_confirm:
                    # Bullish entry: price breaks above R3 + above 4h EMA50
                    if curr_high > r3 and curr_close > curr_ema_50_4h:
                        signals[i] = 0.20
                        position = 1
                    # Bearish entry: price breaks below S3 + below 4h EMA50
                    elif curr_low < s3 and curr_close < curr_ema_50_4h:
                        signals[i] = -0.20
                        position = -1
        
        elif position == 1:  # Long position
            # Exit: price breaks below S3
            if not (np.isnan(prev_high) or np.isnan(prev_low) or np.isnan(prev_close)):
                range_val = prev_high - prev_low
                s3 = prev_close - (range_val * 1.1 / 4)
                if curr_low < s3:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price breaks above R3
            if not (np.isnan(prev_high) or np.isnan(prev_low) or np.isnan(prev_close)):
                range_val = prev_high - prev_low
                r3 = prev_close + (range_val * 1.1 / 4)
                if curr_high > r3:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
            else:
                signals[i] = -0.20
    
    return signals