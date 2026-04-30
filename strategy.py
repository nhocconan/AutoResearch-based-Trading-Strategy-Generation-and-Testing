#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume confirmation
# Uses Camarilla pivot levels from 1d for structure (proven edge on ETH and SOL)
# Only trade breakouts above R3 or below S3 in direction of 1d EMA50 trend
# Volume spike (2.0x 20-period average) confirms institutional participation
# 1d EMA50 provides smoother trend than shorter EMAs, reducing whipsaw in ranging markets
# Discrete sizing 0.25 minimizes fee churn. Target: 50-150 total trades over 4 years (12-37/year).
# Works in both bull and bear markets by following the 1d EMA50 trend direction.

name = "12h_Camarilla_R3S3_Breakout_1dEMA50_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop (MTF Rule #1)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for volume MA calculation
    
    for i in range(start_idx, n):
        # Need prior 1d OHLC for Camarilla calculation (lookback 1 day)
        if i < 1:  # Need at least one prior bar for 1d OHLC (approximation for 12h TF)
            signals[i] = 0.0
            continue
            
        # Approximate prior 1d OHLC using last 2 12h bars (since 2*12h = 24h ~ 1d)
        # For more precision, we would use actual 1d data from df_1d
        # But for simplicity and to avoid look-ahead, we use the most recent completed 1d bar
        # We'll get the prior 1d bar's OHLC from df_1d aligned to current time
        
        # Instead, let's use the actual 1d data for Camarilla calculation
        # We need the prior completed 1d bar's OHLC
        # Since we're on 12h TF, we can use the 1d data directly
        
        # Find the index in df_1d that corresponds to the prior completed 1d bar
        # We'll use the aligned 1d data to get the prior bar's values
        
        # Simpler approach: calculate Camarilla levels using the prior 1d bar's OHLC
        # We'll shift the 1d OHLC by 1 bar to avoid look-ahead
        
        # For now, let's use a different approach: use the current 1d bar's OHLC but with a 1-bar delay
        # This is acceptable because we're using completed 1d bar data
        
        # Get the 1d OHLC series
        o_1d = df_1d['open'].values
        h_1d = df_1d['high'].values
        l_1d = df_1d['low'].values
        c_1d = df_1d['close'].values
        
        # Shift by 1 to use prior completed 1d bar (no look-ahead)
        o_1d_shifted = np.roll(o_1d, 1)
        h_1d_shifted = np.roll(h_1d, 1)
        l_1d_shifted = np.roll(l_1d, 1)
        c_1d_shifted = np.roll(c_1d, 1)
        # Set first value to NaN since we rolled
        o_1d_shifted[0] = np.nan
        h_1d_shifted[0] = np.nan
        l_1d_shifted[0] = np.nan
        c_1d_shifted[0] = np.nan
        
        # Align the shifted 1d OHLC to 12h timeframe
        o_1d_aligned = align_htf_to_ltf(prices, df_1d, o_1d_shifted)
        h_1d_aligned = align_htf_to_ltf(prices, df_1d, h_1d_shifted)
        l_1d_aligned = align_htf_to_ltf(prices, df_1d, l_1d_shifted)
        c_1d_aligned = align_htf_to_ltf(prices, df_1d, c_1d_shifted)
        
        # Calculate Camarilla levels for prior 1d bar
        # R3 = C + (H-L)*1.1/4
        # S3 = C - (H-L)*1.1/4
        prior_high = h_1d_aligned[i]
        prior_low = l_1d_aligned[i]
        prior_close = c_1d_aligned[i]
        
        if np.isnan(prior_high) or np.isnan(prior_low) or np.isnan(prior_close):
            signals[i] = 0.0
            continue
            
        camarilla_range = prior_high - prior_low
        r3 = prior_close + camarilla_range * 1.1 / 4
        s3 = prior_close - camarilla_range * 1.1 / 4
        
        # Volume confirmation: volume > 2.0x 20-period average
        vol_ma_20 = np.mean(volume[max(0, i-20):i])
        volume_spike = volume[i] > (2.0 * vol_ma_20)
        
        curr_close = close[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if volume_spike:
                # Bullish entry: price breaks above R3 AND above 1d EMA50 (uptrend)
                if curr_close > r3 and curr_close > curr_ema_50_1d:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: price breaks below S3 AND below 1d EMA50 (downtrend)
                elif curr_close < s3 and curr_close < curr_ema_50_1d:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when price falls below S3 or below 1d EMA50
            if curr_close < s3 or curr_close < curr_ema_50_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price rises above R3 or above 1d EMA50
            if curr_close > r3 or curr_close > curr_ema_50_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals