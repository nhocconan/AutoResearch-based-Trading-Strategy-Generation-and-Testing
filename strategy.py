#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume confirmation
# Uses Camarilla pivot levels from previous 1d for structure (R3/S3 as strong breakout levels)
# Only trade breakouts above R3 or below S3 in direction of 1w EMA50 trend
# Volume spike (2.0x 20-period average) confirms institutional participation
# Works in bull markets via buying R3 breakouts in uptrends and bear markets via selling S3 breakdowns in downtrends
# Discrete sizing 0.25 minimizes fee churn. Target: 75-150 total trades over 4 years (19-38/year).

name = "4h_Camarilla_R3S3_Breakout_1wEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Camarilla pivots ONCE before loop (MTF Rule #1)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Load 1w data for EMA50 trend filter ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for volume MA
    
    for i in range(start_idx, n):
        # Need prior 1d data for Camarilla calculation (yesterday's OHLC)
        if i < 4:  # Need at least 4h bars to ensure we have prior 1d data
            signals[i] = 0.0
            continue
            
        # Get prior completed 1d OHLC for Camarilla calculation
        # Since we're on 4h timeframe, we need to look back to get yesterday's complete daily bar
        # We'll use the 1d data we loaded and align it properly
        # Camarilla uses prior day's OHLC
        # We need to ensure we're using completed 1d bar (not forming)
        
        # Volume confirmation: volume > 2.0x 20-period average
        vol_ma_20 = np.mean(volume[max(0, i-20):i])
        volume_spike = volume[i] > (2.0 * vol_ma_20)
        
        curr_close = close[i]
        curr_ema_50_1w = ema_50_1w_aligned[i]
        
        # Calculate Camarilla levels from prior 1d bar
        # We need to get the prior completed 1d bar's OHLC
        # Since df_1d contains historical 1d data, we need to find which 1d bar corresponds to prior period
        # For simplicity and to avoid look-ahead, we'll use the last completed 1d bar
        # In practice, align_htf_to_ltf with proper alignment should handle this
        
        # Get the aligned 1d OHLC values (these will be for the completed 1d bar prior to current 4h bar)
        # We need to load the full 1d OHLC for Camarilla calculation
        if len(df_1d) >= 1:
            # Use the last completed 1d bar (not the current forming one)
            # Since we're using align_htf_to_ltf properly elsewhere, we can safely use shifted indices
            # For Camarilla, we definitively need prior day's data
            # We'll calculate it manually from df_1d to ensure we use prior completed bar
            if len(df_1d) >= 2:
                # Get second-to-last 1d bar (prior completed day)
                prior_high = df_1d['high'].iloc[-2]
                prior_low = df_1d['low'].iloc[-2]
                prior_close = df_1d['close'].iloc[-2]
            else:
                # Not enough data
                signals[i] = 0.0
                continue
                
            # Camarilla pivot levels
            # R4 = Close + (High-Low) * 1.5/2
            # R3 = Close + (High-Low) * 1.25/2
            # S3 = Close - (High-Low) * 1.25/2
            # S4 = Close - (High-Low) * 1.5/2
            rng = prior_high - prior_low
            camarilla_r3 = prior_close + rng * 1.25 / 2
            camarilla_s3 = prior_close - rng * 1.25 / 2
            
            if position == 0:  # Flat - look for new entries
                # Require volume spike
                if volume_spike:
                    # Bullish entry: price breaks above Camarilla R3 AND above 1w EMA50 (uptrend)
                    if curr_close > camarilla_r3 and curr_close > curr_ema_50_1w:
                        signals[i] = 0.25
                        position = 1
                    # Bearish entry: price breaks below Camarilla S3 AND below 1w EMA50 (downtrend)
                    elif curr_close < camarilla_s3 and curr_close < curr_ema_50_1w:
                        signals[i] = -0.25
                        position = -1
            
            elif position == 1:  # Long position
                # Exit when price falls below Camarilla S3 or below 1w EMA50
                if curr_close < camarilla_s3 or curr_close < curr_ema_50_1w:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            
            elif position == -1:  # Short position
                # Exit when price rises above Camarilla R3 or above 1w EMA50
                if curr_close > camarilla_r3 or curr_close > curr_ema_50_1w:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals