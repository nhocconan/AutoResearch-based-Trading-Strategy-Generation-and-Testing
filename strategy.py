#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1w EMA50 trend filter and volume confirmation
# Williams Alligator uses three smoothed Moving Averages (Jaw, Teeth, Lips) to identify trends.
# Jaw (13-period, 8-bar shift), Teeth (8-period, 5-bar shift), Lips (5-period, 3-bar shift).
# Trend is bullish when Lips > Teeth > Jaw, bearish when Lips < Teeth < Jaw.
# 1w EMA50 as higher timeframe trend filter (bullish when price > EMA50, bearish when price < EMA50).
# Volume spike (1.8x 20-period average) confirms breakout strength.
# Discrete sizing 0.25 minimizes fee churn. Works in bull via Alligator longs with uptrend,
# in bear via Alligator shorts with downtrend. Target: 12-37 trades/year (50-150 total over 4 years).

name = "12h_WilliamsAlligator_1wEMA50_VolumeSpike_v1"
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
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Williams Alligator components (12h timeframe)
    # Jaw: 13-period SMMA, shifted 8 bars
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)
    jaw[:8] = np.nan  # First 8 values invalid due to shift
    
    # Teeth: 8-period SMMA, shifted 5 bars
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)
    teeth[:5] = np.nan  # First 5 values invalid due to shift
    
    # Lips: 5-period SMMA, shifted 3 bars
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips, 3)
    lips[:3] = np.nan  # First 3 values invalid due to shift
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(20, 13, 8, 5, 50)  # warmup for volume MA, Alligator, and 1w EMA50
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
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
        curr_ema_50 = ema_50_1w_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if curr_volume_spike:
                # Bullish entry: Lips > Teeth > Jaw (bullish alignment) AND price > 1w EMA50 (uptrend)
                if curr_lips > curr_teeth > curr_jaw and curr_close > curr_ema_50:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: Lips < Teeth < Jaw (bearish alignment) AND price < 1w EMA50 (downtrend)
                elif curr_lips < curr_teeth < curr_jaw and curr_close < curr_ema_50:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when Alligator alignment turns bearish (Lips < Teeth OR Teeth < Jaw) OR price drops below Jaw
            if curr_lips < curr_teeth or curr_teeth < curr_jaw or curr_close < curr_jaw:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when Alligator alignment turns bullish (Lips > Teeth OR Teeth > Jaw) OR price rises above Jaw
            if curr_lips > curr_teeth or curr_teeth > curr_jaw or curr_close > curr_jaw:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals