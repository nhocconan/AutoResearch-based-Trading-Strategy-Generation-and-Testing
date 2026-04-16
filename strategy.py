#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion with 1d EMA50 trend filter and volume spike confirmation.
# Long when Williams %R < -80 (oversold) AND price > 1d EMA50 (uptrend) AND 12h volume > 1.5x 20-period average.
# Short when Williams %R > -20 (overbought) AND price < 1d EMA50 (downtrend) AND 12h volume > 1.5x 20-period average.
# Uses discrete position size 0.25. Williams %R identifies exhaustion points, 1d EMA50 ensures alignment with higher timeframe trend,
# volume spike confirms participation. Designed to work in both bull (buy dips) and bear (sell rallies) markets.
# Target: 50-150 trades over 4 years (12-37/year) on 12h timeframe to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h Indicators: Williams %R (14-period) ===
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # === 12h Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Get 1d data once before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for EMA50 calculation
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: EMA50 for trend filter ===
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 12h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for EMA, 20 for volume MA, 14 for Williams %R)
    warmup = 60
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        wr = williams_r[i]
        price = close[i]
        ema_1d = ema_50_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Williams %R rises above -50 (exiting oversold) or volume spike ends
            if wr > -50 or not vol_spike:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Williams %R falls below -50 (exiting overbought) or volume spike ends
            if wr < -50 or not vol_spike:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Williams %R < -80 (oversold) AND price > 1d EMA50 (uptrend) AND volume spike
            if wr < -80 and price > ema_1d and vol_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Williams %R > -20 (overbought) AND price < 1d EMA50 (downtrend) AND volume spike
            elif wr > -20 and price < ema_1d and vol_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "12h_WilliamsR_MeanReversion_1dEMA50_VolumeSpike_V1"
timeframe = "12h"
leverage = 1.0