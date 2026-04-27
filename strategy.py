#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band width regime filter with 12h EMA trend and volume confirmation
# Uses Bollinger Band width to detect low volatility (squeeze) conditions, then breaks out in the direction
# of the 12h EMA trend with volume confirmation. Works in both bull and bear markets by adapting to
# volatility regimes - squeezes often precede significant moves regardless of direction.
# Target: 20-40 trades/year to minimize fee decay while capturing explosive moves after low volatility

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Bollinger Bands (primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 30:
        return np.zeros(n)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 20-period Bollinger Bands on 6h
    close_6h = df_6h['close'].values
    bb_length = 20
    bb_mult = 2.0
    
    # Calculate SMA and standard deviation
    sma_6h = np.full(len(close_6h), np.nan)
    std_dev_6h = np.full(len(close_6h), np.nan)
    
    for i in range(bb_length - 1, len(close_6h)):
        sma_6h[i] = np.mean(close_6h[i-bb_length+1:i+1])
        std_dev_6h[i] = np.std(close_6h[i-bb_length+1:i+1])
    
    # Calculate upper and lower bands
    bb_upper_6h = sma_6h + (bb_mult * std_dev_6h)
    bb_lower_6h = sma_6h - (bb_mult * std_dev_6h)
    bb_width_6h = bb_upper_6h - bb_lower_6h
    
    # Calculate Bollinger Band width percentile (20-period lookback)
    bb_width_percentile = np.full(len(bb_width_6h), np.nan)
    lookback = 20
    for i in range(lookback, len(bb_width_6h)):
        if not np.isnan(bb_width_6h[i-lookback:i+1]).any():
            current_width = bb_width_6h[i]
            historical_widths = bb_width_6h[i-lookback:i+1]
            # Calculate percentile rank of current width
            bb_width_percentile[i] = (np.sum(historical_widths <= current_width) / len(historical_widths)) * 100
    
    # Calculate 12-period EMA on 12h for trend filter
    close_12h = df_12h['close'].values
    ema_len = 12
    ema_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= ema_len:
        multiplier = 2 / (ema_len + 1)
        ema_12h[ema_len-1] = np.mean(close_12h[:ema_len])
        for i in range(ema_len, len(close_12h)):
            ema_12h[i] = (close_12h[i] * multiplier) + (ema_12h[i-1] * (1 - multiplier))
    
    # Calculate average volume on 6h for spike detection
    vol_6h = df_6h['volume'].values
    vol_ma_6h = np.full(len(vol_6h), np.nan)
    vol_period = 6
    for i in range(vol_period, len(vol_6h)):
        vol_ma_6h[i] = np.mean(vol_6h[i-vol_period:i])
    
    # Align all indicators to 6h timeframe
    bb_width_percentile_aligned = align_htf_to_ltf(prices, df_6h, bb_width_percentile)
    bb_upper_6h_aligned = align_htf_to_ltf(prices, df_6h, bb_upper_6h)
    bb_lower_6h_aligned = align_htf_to_ltf(prices, df_6h, bb_lower_6h)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    vol_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_6h)
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(30, 50) + 10
    
    for i in range(start_idx, n):
        if (np.isnan(bb_width_percentile_aligned[i]) or 
            np.isnan(bb_upper_6h_aligned[i]) or 
            np.isnan(bb_lower_6h_aligned[i]) or 
            np.isnan(ema_12h_aligned[i]) or 
            np.isnan(vol_ma_6h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_6h_aligned[i] if vol_ma_6h_aligned[i] > 0 else 0
        
        # Volatility squeeze condition: BB width in lowest 20% of recent range
        volatility_squeeze = bb_width_percentile_aligned[i] < 20
        
        # Volume confirmation: at least 1.5x average volume
        volume_confirmation = vol_ratio > 1.5
        
        if position == 0:
            # Long: Bollinger band breakout above upper band with low volatility and uptrend
            if price > bb_upper_6h_aligned[i] and volatility_squeeze and price > ema_12h_aligned[i] and volume_confirmation:
                signals[i] = size
                position = 1
            # Short: Bollinger band breakout below lower band with low volatility and downtrend
            elif price < bb_lower_6h_aligned[i] and volatility_squeeze and price < ema_12h_aligned[i] and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price closes below Bollinger middle band or volatility expands significantly
            bb_middle_6h_aligned = (bb_upper_6h_aligned[i] + bb_lower_6h_aligned[i]) / 2
            volatility_expansion = bb_width_percentile_aligned[i] > 80
            if price < bb_middle_6h_aligned or volatility_expansion:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price closes above Bollinger middle band or volatility expands significantly
            bb_middle_6h_aligned = (bb_upper_6h_aligned[i] + bb_lower_6h_aligned[i]) / 2
            volatility_expansion = bb_width_percentile_aligned[i] > 80
            if price > bb_middle_6h_aligned or volatility_expansion:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Bollinger_Width_Squeeze_12hEMA_Volume"
timeframe = "6h"
leverage = 1.0