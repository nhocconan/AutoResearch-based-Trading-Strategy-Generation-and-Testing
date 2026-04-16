#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 1w trend filter and volume confirmation.
# Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13.
# Long when Bull Power > 0 AND Bear Power < previous Bear Power (bullish momentum) AND 1w EMA34 uptrend (price > EMA34) AND volume > 1.2x 20-period average.
# Short when Bear Power < 0 AND Bull Power < previous Bull Power (bearish momentum) AND 1w EMA34 downtrend (price < EMA34) AND volume > 1.2x 20-period average.
# Uses discrete position size 0.25. Elder Ray captures momentum behind price moves, 1w EMA34 ensures alignment with higher timeframe trend,
# volume confirmation reduces false signals. Designed to work in both bull (buy strength) and bear (sell weakness) markets.
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
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # === 6h Indicators: Volume Confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.2 * vol_ma)
    
    # Get 1w data once before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:  # Need enough for EMA34 calculation
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # === 1w Indicators: EMA34 for trend filter ===
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w EMA34 to 6h timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 34 periods needed for EMA, 20 for volume MA, 13 for EMA13)
    warmup = 40
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        bp = bull_power[i]
        br = bear_power[i]
        price = close[i]
        ema_1w = ema_34_1w_aligned[i]
        vol_conf = volume_confirm[i]
        
        # Previous bar values for momentum
        if i > 0:
            bp_prev = bull_power[i-1]
            br_prev = bear_power[i-1]
        else:
            bp_prev = bp
            br_prev = br
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if bull power turns negative or volume confirmation ends
            if bp <= 0 or not vol_conf:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if bear power turns positive or volume confirmation ends
            if br >= 0 or not vol_conf:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Bull Power > 0 AND Bull Power increasing (bp > bp_prev) AND price > 1w EMA34 (uptrend) AND volume confirmation
            if bp > 0 and bp > bp_prev and price > ema_1w and vol_conf:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Bear Power < 0 AND Bear Power decreasing (br < br_prev) AND price < 1w EMA34 (downtrend) AND volume confirmation
            elif br < 0 and br < br_prev and price < ema_1w and vol_conf:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_ElderRay_1wEMA34_VolumeConfirm_V1"
timeframe = "6h"
leverage = 1.0