#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 12h EMA50 trend filter and volume confirmation.
# Bull Power = High - EMA13 (measures bull strength), Bear Power = EMA13 - Low (measures bear strength).
# In bull regime (price > 12h EMA50), go long when Bull Power > 0 and rising with volume spike.
# In bear regime (price < 12h EMA50), go short when Bear Power > 0 and rising with volume spike.
# Uses 12h EMA50 for regime filter to avoid counter-trend trades, and Elder Ray to measure
# momentum strength within the regime. Designed for 50-150 total trades over 4 years.

name = "6h_ElderRay_12hEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h EMA50 trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Elder Ray components: EMA13 of close, then Bull/Bear Power
    ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    bull_power = high - ema_13  # measures bull strength: high above EMA13
    bear_power = ema_13 - low   # measures bear strength: EMA13 above low
    
    # Calculate rising momentum: current power > previous power
    bull_power_rising = bull_power > np.roll(bull_power, 1)
    bear_power_rising = bear_power > np.roll(bear_power, 1)
    # Handle first bar
    bull_power_rising[0] = False
    bear_power_rising[0] = False
    
    # Calculate volume regime: current 6h volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get current values
        close_val = close[i]
        ema_trend = ema_50_12h_aligned[i]
        bp = bull_power[i]
        bp_rising = bull_power_rising[i]
        br = bear_power[i]
        br_rising = bear_power_rising[i]
        vol_spike = volume_spike[i]
        
        # Skip if any value is NaN
        if np.isnan(ema_trend) or np.isnan(bp) or np.isnan(br):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Determine regime: bull if close > 12h EMA50, bear if close < 12h EMA50
        is_bull_regime = close_val > ema_trend
        is_bear_regime = close_val < ema_trend
        
        # Regime-based entry conditions
        if is_bull_regime:
            # Long: Bull Power > 0 and rising with volume spike
            long_entry = (bp > 0) and bp_rising and vol_spike
        else:
            long_entry = False
            
        if is_bear_regime:
            # Short: Bear Power > 0 and rising with volume spike
            short_entry = (br > 0) and br_rising and vol_spike
        else:
            short_entry = False
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit on Bull Power <= 0 (loss of bullish momentum) or regime change to bear
            if bp <= 0 or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit on Bear Power <= 0 (loss of bearish momentum) or regime change to bull
            if br <= 0 or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals