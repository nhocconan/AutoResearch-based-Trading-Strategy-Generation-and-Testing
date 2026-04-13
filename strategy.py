#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Bollinger Band squeeze breakout with 1d volume confirmation
    # Long: price breaks above upper BB during low volatility (BB width < 20th percentile) + 1d volume > 1.5x 20-day average
    # Short: price breaks below lower BB during low volatility + 1d volume > 1.5x 20-day average
    # Uses Bollinger Band width percentile to identify squeeze conditions
    # Target: 12-37 trades/year to stay within 12h optimal range (50-150 total over 4 years)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Bollinger Bands and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1-day Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2.0
    
    sma_20 = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_20 = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).std().values
    
    upper_bb = sma_20 + bb_std * std_20
    lower_bb = sma_20 - bb_std * std_20
    
    # Calculate Bollinger Band width and its percentile for squeeze detection
    bb_width = (upper_bb - lower_bb) / sma_20
    # Calculate 50-day percentile of BB width (using 50-day lookback)
    bb_width_percentile = np.zeros_like(bb_width)
    for i in range(50, len(bb_width)):
        if i >= 50:
            bb_width_percentile[i] = np.percentile(bb_width[i-50:i], 20)  # 20th percentile
    
    # Squeeze condition: BB width below 20th percentile of its 50-day range
    squeeze_condition = bb_width < bb_width_percentile
    
    # Calculate 1-day volume average for confirmation
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 12h timeframe
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    squeeze_aligned = align_htf_to_ltf(prices, df_1d, squeeze_condition)
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    # Track entry price for stoploss
    entry_price = np.full(n, np.nan)
    
    # Calculate ATR using true range approximation for 12h timeframe
    atr_12h = np.zeros(n)
    for i in range(1, n):
        tr = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
        if i < 14:
            atr_12h[i] = tr  # Simple average for warmup
        else:
            atr_12h[i] = 0.93 * atr_12h[i-1] + 0.07 * tr  # Wilder's smoothing
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(upper_bb_aligned[i]) or 
            np.isnan(lower_bb_aligned[i]) or
            np.isnan(squeeze_aligned[i]) or
            np.isnan(vol_avg_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-day average
        # Note: We need to check if we have a complete 1d bar for volume
        # Since we're on 12h timeframe, we check the volume of the most recent completed 1d bar
        volume_confirmed = volume_1d[-1] > 1.5 * vol_avg_20_1d_aligned[i] if len(volume_1d) > 0 else False
        
        # Breakout conditions: price breaks Bollinger Bands during squeeze with volume confirmation
        breakout_long = squeeze_aligned[i] and (close[i] > upper_bb_aligned[i]) and volume_confirmed
        breakout_short = squeeze_aligned[i] and (close[i] < lower_bb_aligned[i]) and volume_confirmed
        
        # Stoploss: 2x ATR below/above entry
        exit_long = position == 1 and not np.isnan(entry_price[i-1]) and close[i] < entry_price[i-1] - 2.0 * atr_12h[i]
        exit_short = position == -1 and not np.isnan(entry_price[i-1]) and close[i] > entry_price[i-1] + 2.0 * atr_12h[i]
        
        # Execute signals
        if breakout_long and position != 1:
            position = 1
            signals[i] = position_size
            entry_price[i] = close[i]
        elif breakout_short and position != -1:
            position = -1
            signals[i] = -position_size
            entry_price[i] = close[i]
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
            entry_price[i] = np.nan
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
            entry_price[i] = np.nan
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            elif position == -1:
                signals[i] = -position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            else:
                signals[i] = 0.0
                entry_price[i] = np.nan
    
    return signals

name = "12h_1d_bb_squeeze_breakout_v1"
timeframe = "12h"
leverage = 1.0