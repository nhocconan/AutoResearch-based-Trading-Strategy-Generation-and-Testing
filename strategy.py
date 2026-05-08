#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Bollinger Band Squeeze + 4h Trend Filter + Volume Spike
# Uses 4h EMA50 for trend direction, Bollinger Band width percentile to detect low volatility squeeze,
# and volume spike (>1.5x average) for entry timing. Designed to catch breakouts from low volatility
# periods in both bull and bear markets by following the 4h trend while avoiding choppy conditions.
# Target: 20-40 trades/year.

name = "1h_BollingerSqueeze_4hEMA50_VolumeSpike"
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
    
    # Get 4h data for trend filter and Bollinger Bands
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema50_4h = np.full(len(close_4h), np.nan)
    if len(close_4h) >= 50:
        ema50_4h[49] = np.mean(close_4h[:50])
        for i in range(50, len(close_4h)):
            ema50_4h[i] = (close_4h[i] * 2 + ema50_4h[i-1] * 48) / 50
    
    # Calculate 4h Bollinger Bands (20, 2)
    bb_length = 20
    bb_mult = 2.0
    if len(close_4h) >= bb_length:
        # Calculate rolling mean and std
        bb_middle = np.full(len(close_4h), np.nan)
        bb_std = np.full(len(close_4h), np.nan)
        
        for i in range(bb_length-1, len(close_4h)):
            bb_middle[i] = np.mean(close_4h[i-bb_length+1:i+1])
            bb_std[i] = np.std(close_4h[i-bb_length+1:i+1])
        
        bb_upper = bb_middle + bb_mult * bb_std
        bb_lower = bb_middle - bb_mult * bb_std
        bb_width = bb_upper - bb_lower
        
        # Calculate Bollinger Band width percentile (50-period lookback)
        bb_width_percentile = np.full(len(bb_width), np.nan)
        if len(bb_width) >= 50:
            for i in range(50, len(bb_width)):
                window = bb_width[i-50:i+1]
                valid_vals = window[~np.isnan(window)]
                if len(valid_vals) > 0:
                    rank = (bb_width[i] > valid_vals).sum()
                    bb_width_percentile[i] = (rank / len(valid_vals)) * 100
    else:
        bb_width_percentile = np.full(len(close_4h), np.nan)
    
    # Calculate 4h volume average for volume spike detection
    vol_4h = df_4h['volume'].values
    vol_avg_20_4h = np.full(len(vol_4h), np.nan)
    if len(vol_4h) >= 20:
        for i in range(20, len(vol_4h)):
            vol_avg_20_4h[i] = np.mean(vol_4h[i-20:i])
    
    # Align 4h indicators to 1h timeframe
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    bb_width_percentile_aligned = align_htf_to_ltf(prices, df_4h, bb_width_percentile)
    vol_avg_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_avg_20_4h)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(bb_width_percentile_aligned[i]) or
            np.isnan(vol_avg_20_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current 1h volume > 1.5x 20-period average of 4h volume
        vol_spike = volume[i] > 1.5 * vol_avg_20_4h_aligned[i]
        
        # Bollinger squeeze condition: BB width percentile < 30 (low volatility)
        squeeze_condition = bb_width_percentile_aligned[i] < 30
        
        if position == 0:
            # Look for entry: breakout from squeeze in direction of 4h trend
            long_condition = (
                close[i] > ema50_4h_aligned[i] and   # price above 4h EMA50 (bullish bias)
                squeeze_condition and                # low volatility squeeze
                vol_spike                            # volume spike for breakout confirmation
            )
            
            short_condition = (
                close[i] < ema50_4h_aligned[i] and   # price below 4h EMA50 (bearish bias)
                squeeze_condition and                # low volatility squeeze
                vol_spike                            # volume spike for breakdown confirmation
            )
            
            if long_condition:
                signals[i] = 0.20
                position = 1
            elif short_condition:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price returns below 4h EMA50 or squeeze ends
            if close[i] < ema50_4h_aligned[i] or bb_width_percentile_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price returns above 4h EMA50 or squeeze ends
            if close[i] > ema50_4h_aligned[i] or bb_width_percentile_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals