#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 1w trend filter and volume confirmation.
# Long when BB width < 20th percentile (squeeze) AND price breaks above upper BB AND 1w EMA50 uptrend AND volume > 1.5x 20-period average.
# Short when BB width < 20th percentile (squeeze) AND price breaks below lower BB AND 1w EMA50 downtrend AND volume > 1.5x 20-period average.
# Uses discrete position size 0.25. Bollinger squeeze identifies low volatility primed for breakout,
# 1w EMA50 ensures alignment with weekly trend, volume confirms breakout validity.
# Designed to work in both bull (buy breakouts) and bear (sell breakdowns) markets.
# Target: 80-160 trades over 4 years (20-40/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: Bollinger Bands (20,2) ===
    close_s = pd.Series(close)
    bb_mid = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_mid  # Normalized width
    
    # === 6h Indicators: BB Width Percentile (20-period lookback) ===
    bb_width_series = pd.Series(bb_width)
    bb_width_pct = bb_width_series.rolling(window=20, min_periods=20).rank(pct=True).values
    
    # === 6h Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Get 1w data once before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 55:  # Need enough for EMA50 calculation
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # === 1w Indicators: EMA50 for trend filter ===
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 6h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for EMA, 20 for BB/volume)
    warmup = 60
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(bb_width_pct[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        width_pct = bb_width_pct[i]
        price = close[i]
        bb_up = bb_upper[i]
        bb_low = bb_lower[i]
        ema_1w = ema_50_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price re-enters BB (breakout failed) or volume spike ends
            if price < bb_mid[i] or not vol_spike:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price re-enters BB (breakdown failed) or volume spike ends
            if price > bb_mid[i] or not vol_spike:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: BB squeeze (width < 20th percentile) AND price breaks above upper BB AND 1w EMA50 uptrend AND volume spike
            if width_pct < 0.2 and price > bb_up and price > ema_1w and vol_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT: BB squeeze (width < 20th percentile) AND price breaks below lower BB AND 1w EMA50 downtrend AND volume spike
            elif width_pct < 0.2 and price < bb_low and price < ema_1w and vol_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_BB_Squeeze_Breakout_1wEMA50_VolumeSpike_V1"
timeframe = "6h"
leverage = 1.0