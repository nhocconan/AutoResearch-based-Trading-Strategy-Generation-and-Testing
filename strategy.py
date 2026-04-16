#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 12h trend filter and volume confirmation.
# Long when Bull Power > 0 (close > EMA13) AND Bear Power < 0 (low < EMA13) AND 12h EMA50 uptrend (price > EMA50) AND 6h volume > 1.3x 20-period average.
# Short when Bear Power < 0 (low < EMA13) AND Bull Power < 0 (close < EMA13) AND 12h EMA50 downtrend (price < EMA50) AND 6h volume > 1.3x 20-period average.
# Uses discrete position size 0.25. Elder Ray measures bull/bear strength via EMA13, 12h EMA50 ensures higher timeframe alignment,
# volume spike confirms participation. Designed to work in both bull (buy strength) and bear (sell weakness) markets.
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
    
    # Bull Power = Close - EMA13
    bull_power = close - ema_13
    # Bear Power = Low - EMA13
    bear_power = low - ema_13
    
    # === 6h Indicators: Volume Spike (volume > 1.3x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.3 * vol_ma)
    
    # Get 12h data once before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need enough for EMA50 calculation
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # === 12h Indicators: EMA50 for trend filter ===
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 6h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for EMA50, 20 for volume MA)
    warmup = 60
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        bp = bull_power[i]
        br = bear_power[i]
        price = close[i]
        ema_12h = ema_50_12h_aligned[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Bull Power turns negative OR Bear Power turns positive OR volume spike ends
            if bp <= 0 or br >= 0 or not vol_spike:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Bull Power turns positive OR Bear Power turns negative OR volume spike ends
            if bp >= 0 or br <= 0 or not vol_spike:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Bull Power > 0 AND Bear Power < 0 AND price > 12h EMA50 (uptrend) AND volume spike
            if bp > 0 and br < 0 and price > ema_12h and vol_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Bull Power < 0 AND Bear Power > 0 AND price < 12h EMA50 (downtrend) AND volume spike
            elif bp < 0 and br > 0 and price < ema_12h and vol_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_ElderRay_12hEMA50_VolumeSpike_V1"
timeframe = "6h"
leverage = 1.0