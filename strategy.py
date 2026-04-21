#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with volume confirmation and 12h trend filter.
# Go long when price breaks above Donchian(20) high and volume > 1.5x 20-period average.
# Go short when price breaks below Donchian(20) low and volume > 1.5x 20-period average.
# Only take trades in direction of 12h EMA50 trend (long when price > EMA50, short when price < EMA50).
# Uses 12h EMA50 for trend filter to avoid counter-trend trades.
# Target: 20-50 trades/year by requiring trend alignment + breakout + volume confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 12h for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 4h
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if np.isnan(ema50_12h_aligned[i]):
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
        
        # Trend filter: price vs 12h EMA50
        bull_trend = price > ema50_12h_aligned[i]
        bear_trend = price < ema50_12h_aligned[i]
        
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

name = "4h_Donchian_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0