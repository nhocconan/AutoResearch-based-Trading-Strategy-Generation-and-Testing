#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w EMA50 trend filter and volume spike confirmation
# Williams Alligator (Jaw/Teeth/Lips) identifies trend via smoothed moving averages convergence/divergence
# Breakout occurs when Lips cross above/below Teeth with Jaw positioned for trend direction
# 1w EMA50 ensures alignment with weekly trend to avoid counter-trend trades
# Volume spike (2.0x 24-period average) confirms institutional participation
# Discrete sizing 0.25 minimizes fee churn. Target: 30-100 total trades over 4 years (7-25/year).

name = "1d_WilliamsAlligator_1wEMA50_VolumeSpike_v1"
timeframe = "1d"
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
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Williams Alligator on 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    median_price_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3.0
    
    # Jaw (blue): 13-period SMMA, shifted 8 bars
    jaw = pd.Series(median_price_1d).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)
    jaw[:8] = np.nan
    
    # Teeth (red): 8-period SMMA, shifted 5 bars
    teeth = pd.Series(median_price_1d).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)
    teeth[:5] = np.nan
    
    # Lips (green): 5-period SMMA, shifted 3 bars
    lips = pd.Series(median_price_1d).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips, 3)
    lips[:3] = np.nan
    
    # Align to 1d timeframe (no additional delay needed as Alligator uses completed bars)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Volume confirmation: volume > 2.0x 24-period average (24 days)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(50, 13, 8, 5, 8) + 8  # warmup for all indicators plus Jaw shift
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_lips = lips_aligned[i]
        curr_teeth = teeth_aligned[i]
        curr_jaw = jaw_aligned[i]
        curr_ema_50 = ema_50_1w_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if curr_volume_spike:
                # Bullish entry: Lips cross above Teeth AND Lips > Jaw (uptrend alignment)
                if curr_lips > curr_teeth and curr_lips > curr_jaw and curr_close > curr_ema_50:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: Lips cross below Teeth AND Lips < Jaw (downtrend alignment)
                elif curr_lips < curr_teeth and curr_lips < curr_jaw and curr_close < curr_ema_50:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when Lips cross below Teeth (trend weakness) OR price drops below 1w EMA50
            if curr_lips < curr_teeth or curr_close < curr_ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when Lips cross above Teeth (trend weakness) OR price rises above 1w EMA50
            if curr_lips > curr_teeth or curr_close > curr_ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals