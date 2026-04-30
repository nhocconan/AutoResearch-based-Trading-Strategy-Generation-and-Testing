#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume spike confirmation
# Williams Alligator: Jaw (EMA13, 8-period shift), Teeth (EMA8, 5-period shift), Lips (EMA5, 3-period shift)
# Bullish when Lips > Teeth > Jaw (aligned), Bearish when Lips < Teeth < Jaw (aligned)
# 1d EMA50 as trend filter: only long when price > EMA50, short when price < EMA50
# Volume spike (2.0x 20-period average) confirms momentum
# Discrete sizing 0.25 minimizes fee churn. Works in bull via Alligator longs with uptrend,
# in bear via Alligator shorts with downtrend. Target: 12-37 trades/year (50-150 total over 4 years).

name = "12h_WilliamsAlligator_1dEMA50_VolumeSpike_v1"
timeframe = "12h"
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
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams Alligator components (12h timeframe)
    # Jaw: Blue line - EMA(13, 8)
    jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)  # shift 8 periods forward
    jaw[:8] = np.nan
    
    # Teeth: Red line - EMA(8, 5)
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)  # shift 5 periods forward
    teeth[:5] = np.nan
    
    # Lips: Green line - EMA(5, 3)
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().values
    lips = np.roll(lips, 3)  # shift 3 periods forward
    lips[:3] = np.nan
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(20, 13, 8, 5, 50)  # warmup for volume MA, Alligator, and 1d EMA50
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_jaw = jaw[i]
        curr_teeth = teeth[i]
        curr_lips = lips[i]
        curr_ema_50 = ema_50_1d_aligned[i]
        curr_volume_spike = volume_spike[i]
        prev_lips = lips[i-1] if i > 0 else 0
        prev_teeth = teeth[i-1] if i > 0 else 0
        prev_jaw = jaw[i-1] if i > 0 else 0
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if curr_volume_spike:
                # Bullish entry: Lips > Teeth > Jaw (Alligator aligned up) AND price > 1d EMA50 (uptrend)
                if curr_lips > curr_teeth and curr_teeth > curr_jaw and curr_close > curr_ema_50:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: Lips < Teeth < Jaw (Alligator aligned down) AND price < 1d EMA50 (downtrend)
                elif curr_lips < curr_teeth and curr_teeth < curr_jaw and curr_close < curr_ema_50:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when Alligator loses alignment (Lips <= Teeth OR Teeth <= Jaw) OR price drops below EMA50
            if curr_lips <= curr_teeth or curr_teeth <= curr_jaw or curr_close < curr_ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when Alligator loses alignment (Lips >= Teeth OR Teeth >= Jaw) OR price rises above EMA50
            if curr_lips >= curr_teeth or curr_teeth >= curr_jaw or curr_close > curr_ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals