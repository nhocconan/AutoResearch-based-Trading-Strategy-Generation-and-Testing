#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1w EMA50 trend filter and volume confirmation.
# Long when Bull Power > 0 (close > EMA13) AND 1w EMA50 uptrend (price > EMA50) AND 6h volume > 1.5x 20-period average.
# Short when Bear Power < 0 (close < EMA13) AND 1w EMA50 downtrend (price < EMA50) AND 6h volume > 1.5x 20-period average.
# Elder Ray measures trend strength via deviation from EMA13. 1w EMA50 ensures alignment with weekly trend.
# Volume spike confirms institutional participation. Works in bull (buy strength) and bear (sell weakness).
# Target: 80-160 trades over 4 years (20-40/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: EMA13 for Elder Ray ===
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = close - ema_13  # >0 = bullish
    bear_power = close - ema_13  # <0 = bearish (same calc, interpret sign)
    
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
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for EMA, 20 for volume MA)
    warmup = 60
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        bp = bull_power[i]  # Bull Power
        price = close[i]
        ema_1w = ema_50_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Bull Power turns negative (trend weakness) or volume spike ends
            if bp <= 0 or not vol_spike:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Bull Power turns non-negative (trend strength) or volume spike ends
            if bp >= 0 or not vol_spike:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Bull Power > 0 (bullish) AND price > 1w EMA50 (uptrend) AND volume spike
            if bp > 0 and price > ema_1w and vol_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Bull Power < 0 (bearish) AND price < 1w EMA50 (downtrend) AND volume spike
            elif bp < 0 and price < ema_1w and vol_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_ElderRay_1wEMA50_VolumeSpike_V1"
timeframe = "6h"
leverage = 1.0