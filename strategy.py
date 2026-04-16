#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R(14) mean reversion with 1w EMA(50) trend filter and volume confirmation.
# Long when Williams %R < -80 AND price > 1w EMA(50) (uptrend) AND volume > 1.3x 20-period average.
# Short when Williams %R > -20 AND price < 1w EMA(50) (downtrend) AND volume > 1.3x 20-period average.
# Uses discrete position size 0.25. Williams %R captures overextended moves, 1w EMA ensures we trade with higher timeframe trend (avoiding counter-trend whipsaws),
# volume spike confirms participation. Designed to work in both bull (buy dips in uptrend) and bear (sell rallies in downtrend) markets.
# Target: 50-150 trades over 4 years (12-37/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h Indicators: Williams %R(14) ===
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r_values = williams_r.values
    
    # === 12h Indicators: Volume Spike (volume > 1.3x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_spike = volume > (1.3 * vol_ma)
    volume_spike_values = volume_spike.values
    
    # Get 1w data once before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:  # Need enough for EMA calculation
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # === 1w Indicators: EMA(50) for trend filter ===
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean()
    ema_50_values = ema_50.values
    
    # Align 1w EMA to 12h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_values)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for EMA, 20 for volume MA, 14 for Williams %R)
    warmup = 60
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_values[i]) or np.isnan(ema_50_aligned[i]) or
            np.isnan(vol_ma.values[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        williams_val = williams_r_values[i]
        ema_50_val = ema_50_aligned[i]
        vol_spike = volume_spike_values[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Williams %R returns to oversold threshold (-50) or volume spike ends
            if williams_val >= -50 or not vol_spike:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Williams %R returns to overbought threshold (-50) or volume spike ends
            if williams_val <= -50 or not vol_spike:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Williams %R < -80 AND price > 1w EMA(50) (uptrend) AND volume spike
            if williams_val < -80 and price > ema_50_val and vol_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Williams %R > -20 AND price < 1w EMA(50) (downtrend) AND volume spike
            elif williams_val > -20 and price < ema_50_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "12h_WilliamsR14_1wEMA50_VolumeSpike_V1"
timeframe = "12h"
leverage = 1.0