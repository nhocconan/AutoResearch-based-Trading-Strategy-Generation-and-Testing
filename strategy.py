#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 12h EMA34 trend filter and volume confirmation.
# Long when Bull Power > 0 AND price > 12h EMA34 (bullish regime) AND volume > 1.5x 20-period average.
# Short when Bear Power < 0 AND price < 12h EMA34 (bearish regime) AND volume > 1.5x 20-period average.
# Elder Ray measures bull/bear strength relative to EMA13. Combined with higher timeframe trend filter (12h EMA34)
# to avoid counter-trend trades, and volume spike to confirm institutional participation.
# Designed to work in both bull (buy strength) and bear (sell weakness) markets by aligning with 12h trend.
# Target: 80-180 trades over 4 years (20-45/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: Elder Ray Index (Bull Power, Bear Power) ===
    # EMA13 for Elder Ray
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    # === 6h Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Get 12h data once before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 40:  # Need enough for EMA34 calculation
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # === 12h Indicators: EMA34 for trend filter ===
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 34 periods needed for EMA34, 20 for volume MA, 13 for EMA13)
    warmup = 40
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema34_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        ema34_val = ema34_12h_aligned[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Bull Power turns negative OR price breaks below 12h EMA34 OR volume spike ends
            if bull_val <= 0 or price < ema34_val or not vol_spike:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Bear Power turns positive OR price breaks above 12h EMA34 OR volume spike ends
            if bear_val >= 0 or price > ema34_val or not vol_spike:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Bull Power > 0 AND price > 12h EMA34 (bullish regime) AND volume spike
            if bull_val > 0 and price > ema34_val and vol_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Bear Power < 0 AND price < 12h EMA34 (bearish regime) AND volume spike
            elif bear_val < 0 and price < ema34_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_ElderRay_12hEMA34_VolumeSpike_V1"
timeframe = "6h"
leverage = 1.0