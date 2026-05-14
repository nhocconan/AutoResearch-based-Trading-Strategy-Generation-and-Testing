#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA50 trend filter and volume spike confirmation.
# Elder Ray measures bull/bear strength relative to EMA. In bull regime (price > 1d EMA50),
# we go long when Bull Power > 0 and volume spikes. In bear regime (price < 1d EMA50),
# we go short when Bear Power < 0 and volume spikes. This adapts to both bull and bear markets
# by using the higher timeframe trend for regime filtering and Elder Ray for precise entry timing.

name = "6h_ElderRay_1dTrend_VolumeSpike_Regime"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate 13-period EMA for Elder Ray (primary timeframe)
    ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema_13  # Bull Power: High - EMA
    bear_power = low - ema_13   # Bear Power: Low - EMA
    
    # Calculate volume regime: current 6h volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get current values
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        ema_trend = ema_50_aligned[i]
        vol_spike = volume_spike[i]
        close_val = close[i]
        
        # Skip if any value is NaN
        if np.isnan(bull_val) or np.isnan(bear_val) or np.isnan(ema_trend):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Determine regime: bull if close > 1d EMA50, bear if close < 1d EMA50
        is_bull_regime = close_val > ema_trend
        is_bear_regime = close_val < ema_trend
        
        # Regime-based entry conditions
        if is_bull_regime:
            # Long: Bull Power > 0 (bulls in control) with volume spike
            long_entry = (bull_val > 0) and vol_spike
        else:
            long_entry = False
            
        if is_bear_regime:
            # Short: Bear Power < 0 (bears in control) with volume spike
            short_entry = (bear_val < 0) and vol_spike
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
            if bull_val <= 0 or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit on Bear Power >= 0 (loss of bearish momentum) or regime change to bull
            if bear_val >= 0 or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals