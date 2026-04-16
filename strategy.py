#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R(14) extreme reversal with 12h EMA34 trend filter and volume confirmation.
# Long when Williams %R crosses above -80 from below AND price > 12h EMA34 (uptrend) AND volume > 1.3x 20-period average.
# Short when Williams %R crosses below -20 from above AND price < 12h EMA34 (downtrend) AND volume > 1.3x 20-period average.
# Uses discrete position size 0.25. Williams %R captures oversold/overbought reversals, 12h EMA34 ensures alignment with higher timeframe trend,
# volume spike confirms participation. Designed to work in both bull (buy reversals from oversold) and bear (sell reversals from overbought) markets.
# Target: 50-150 trades over 4 years (12-38/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: Williams %R(14) ===
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # === 6h Indicators: Volume Spike (volume > 1.3x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.3 * vol_ma)
    
    # Get 12h data once before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:  # Need enough for EMA34 calculation
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # === 12h Indicators: EMA34 for trend filter ===
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 12h EMA34 to 6h timeframe
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 34 periods needed for EMA, 20 for volume MA, 14 for Williams %R)
    warmup = 40
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    prev_williams_r = williams_r[warmup-1] if warmup > 0 else -50
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(williams_r[i]) or np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            prev_williams_r = williams_r[i]
            continue
        
        # Current values
        price = close[i]
        wr = williams_r[i]
        ema_12h = ema_34_12h_aligned[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Williams %R rises above -20 (overbought) or volume spike ends
            if wr > -20 or not vol_spike:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Williams %R falls below -80 (oversold) or volume spike ends
            if wr < -80 or not vol_spike:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            prev_williams_r = wr
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Williams %R crossover conditions
            wr_cross_up = (prev_williams_r <= -80) and (wr > -80)  # Cross above -80
            wr_cross_down = (prev_williams_r >= -20) and (wr < -20)  # Cross below -20
            
            # LONG: Williams %R crosses above -80 from below AND price > 12h EMA34 (uptrend) AND volume spike
            if wr_cross_up and price > ema_12h and vol_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Williams %R crosses below -20 from above AND price < 12h EMA34 (downtrend) AND volume spike
            elif wr_cross_down and price < ema_12h and vol_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
        
        prev_williams_r = wr
    
    return signals

name = "6h_WilliamsR14_12hEMA34_VolumeSpike_V1"
timeframe = "6h"
leverage = 1.0