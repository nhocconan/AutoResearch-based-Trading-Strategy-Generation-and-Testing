#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R extremes with 12h trend filter and volume confirmation
# Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
# Long when Williams %R < -80 (oversold) and 12h EMA(50) rising + volume spike
# Short when Williams %R > -20 (overbought) and 12h EMA(50) falling + volume spike
# Exit when Williams %R returns to -50 level (mean reversion)
# Designed for low trade frequency (19-50/year) to minimize fee drag. Works in both bull and bear markets by fading extremes in the direction of the higher timeframe trend.

name = "4h_WilliamsR_Extremes_12hEMA50_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA(50) trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate EMA(50) on 12h for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA to 4h timeframe (wait for completed 12h bar)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Williams %R on 4h (14-period)
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Volume confirmation (2.0x 20-period average) on 4h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = max(50, period + 5, 20 + 1)  # 12h EMA50 + Williams %R14 + volume MA20 + shift
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Williams %R below -80 (oversold) AND 12h EMA rising
            williams_oversold = williams_r[i] < -80
            ema_rising = ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1]
            
            # Williams %R above -20 (overbought) AND 12h EMA falling
            williams_overbought = williams_r[i] > -20
            ema_falling = ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1]
            
            # Long entry: Oversold + 12h EMA rising + volume spike
            if (williams_oversold and ema_rising and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Overbought + 12h EMA falling + volume spike
            elif (williams_overbought and ema_falling and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R returns to -50 (mean reversion) OR 12h EMA starts falling
            if williams_r[i] >= -50 or ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R returns to -50 (mean reversion) OR 12h EMA starts rising
            if williams_r[i] <= -50 or ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals