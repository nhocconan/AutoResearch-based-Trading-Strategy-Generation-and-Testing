#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with volume confirmation and weekly trend filter.
# Go long when price breaks above Donchian(20) high and volume > 1.5x 20-period average.
# Go short when price breaks below Donchian(20) low and volume > 1.5x 20-period average.
# Only take trades in direction of weekly EMA200 trend (long when price > EMA200, short when price < EMA200).
# Uses weekly EMA200 for trend filter to avoid counter-trend trades.
# Target: 12-37 trades/year by requiring trend alignment + breakout + volume confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # Calculate weekly EMA200 for trend filter
    close_w = df_1w['close'].values
    ema200_w = pd.Series(close_w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align weekly EMA200 to 6h
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_w)
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if np.isnan(ema200_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Donchian channels (20-period)
        lookback_start = max(0, i - 19)
        high_window = prices['high'].iloc[lookback_start:i+1].values
        low_window = prices['low'].iloc[lookback_start:i+1].values
        
        donchian_high = np.max(high_window)
        donchian_low = np.min(low_window)
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Calculate 20-period volume average
        vol_lookback_start = max(0, i - 19)
        vol_window = prices['volume'].iloc[vol_lookback_start:i+1].values
        vol_ma_20 = np.mean(vol_window)
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume > 1.5 * vol_ma_20
        
        # Trend filter: price vs weekly EMA200
        bull_trend = price > ema200_1w_aligned[i]
        bear_trend = price < ema200_1w_aligned[i]
        
        if position == 0:
            # Enter long on breakout above Donchian high with volume and bullish trend
            if price > donchian_high and volume_confirm and bull_trend:
                signals[i] = 0.25
                position = 1
            # Enter short on breakout below Donchian low with volume and bearish trend
            elif price < donchian_low and volume_confirm and bear_trend:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price crosses back through Donchian opposite level
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below Donchian low
                if price < donchian_low:
                    exit_signal = True
            elif position == -1:
                # Exit short: price breaks above Donchian high
                if price > donchian_high:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Donchian_Breakout_Volume_Trend"
timeframe = "6h"
leverage = 1.0