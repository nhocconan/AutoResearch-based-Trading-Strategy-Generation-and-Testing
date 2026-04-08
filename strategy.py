# 4h Donchian breakout with volume confirmation and 1d trend filter
# Designed for low trade frequency (<30/year) to avoid fee drag
# Works in bull/bear by only trading in direction of 1d trend
# Entry: 4h price breaks 20-period 4h Donchian in direction of 1d trend + volume spike
# Exit: opposite Donchian break or trailing stop
# Position size: 0.25 (discrete to minimize churn)

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_trend_filter_volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channels (primary timeframe)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 20-period Donchian channels on 4h data
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Upper band: 20-period high
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    # Lower band: 20-period low
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe (no additional delay needed)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 50-period EMA on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: volume > 1.8x 20-period average (higher threshold = fewer trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_confirm = vol_ratio > 1.8
    
    # Session filter: 08-20 UTC (avoid low-volume Asian session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup period
    start_idx = max(20, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if Donchian levels or EMA not available
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or np.isnan(ema_50_aligned[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Only consider new signals during session with volume confirmation
        if not (in_session[i] and vol_confirm[i]):
            if position != 0:
                # Hold existing position
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below 4h Donchian low (breakdown) OR below 1d EMA (trend change)
            if close[i] < donchian_low_aligned[i] or close[i] < ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price closes above 4h Donchian high (breakout) OR above 1d EMA (trend change)
            if close[i] > donchian_high_aligned[i] or close[i] > ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price breaks above 4h Donchian high with volume AND above 1d EMA (uptrend)
            if close[i] > donchian_high_aligned[i] and close[i] > ema_50_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below 4h Donchian low with volume AND below 1d EMA (downtrend)
            elif close[i] < donchian_low_aligned[i] and close[i] < ema_50_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals