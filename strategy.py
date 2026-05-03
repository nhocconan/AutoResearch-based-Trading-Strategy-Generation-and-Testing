#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 12h EMA50 trend filter and volume confirmation.
# Bull Power = High - EMA13 (12h), Bear Power = EMA13 - Low (12h).
# Long when Bull Power > 0 AND increasing AND price > 12h EMA50 (bull trend) AND volume spike.
# Short when Bear Power > 0 AND increasing AND price < 12h EMA50 (bear trend) AND volume spike.
# Uses 12h for trend/power calculation to reduce noise vs 6h, volume spike on 6h for confirmation.
# Designed for 50-150 total trades over 4 years (12-37/year). Works in both bull and bear regimes
# by adapting to trend direction via 12h EMA50. Focus on BTC/ETH as primary symbols.

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
    
    # Get 12h data for Elder Ray and EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA13 for Elder Ray power
    ema_13_12h = pd.Series(close_12h).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power = high_12h - ema_13_12h
    bear_power = ema_13_12h - low_12h
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align 12h indicators to 6h (wait for 12h bar to complete)
    bull_power_aligned = align_htf_to_ltf(prices, df_12h, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_12h, bear_power)
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate volume regime: current 6h volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get current values
        close_val = close[i]
        bull_pow = bull_power_aligned[i]
        bear_pow = bear_power_aligned[i]
        ema_trend = ema_50_aligned[i]
        vol_spike = volume_spike[i]
        
        # Skip if any value is NaN
        if np.isnan(bull_pow) or np.isnan(bear_pow) or np.isnan(ema_trend):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Calculate power change (current - previous) for momentum confirmation
        if i > 100:
            bull_pow_change = bull_pow - bull_power_aligned[i-1]
            bear_pow_change = bear_pow - bear_power_aligned[i-1]
        else:
            bull_pow_change = 0
            bear_pow_change = 0
        
        # Determine regime: bull if close > 12h EMA50, bear if close < 12h EMA50
        is_bull_regime = close_val > ema_trend
        is_bear_regime = close_val < ema_trend
        
        # Regime-based entry conditions with power momentum
        if is_bull_regime:
            # Long: Bull Power > 0 AND increasing (rising momentum) in bull trend AND volume spike
            long_entry = (bull_pow > 0) and (bull_pow_change > 0) and vol_spike
        else:
            long_entry = False
            
        if is_bear_regime:
            # Short: Bear Power > 0 AND increasing (rising momentum) in bear trend AND volume spike
            short_entry = (bear_pow > 0) and (bear_pow_change > 0) and vol_spike
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
            if bull_pow <= 0 or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit on Bear Power <= 0 (loss of bearish momentum) or regime change to bull
            if bear_pow <= 0 or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals