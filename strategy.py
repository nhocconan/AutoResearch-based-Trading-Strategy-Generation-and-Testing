#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Williams %R extremes with 1w EMA50 trend filter and volume confirmation.
# Long when Williams %R < -80 (oversold) AND price > 1w EMA50 (uptrend) AND 12h volume > 1.5x 20-period average.
# Short when Williams %R > -20 (overbought) AND price < 1w EMA50 (downtrend) AND 12h volume > 1.5x 20-period average.
# Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts).
# Uses discrete position size 0.25. 1d/1w filters provide signal direction, 12h provides entry timing.
# Target: 50-150 total trades over 4 years (12-37/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 1w data once before loop for EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # === 1d Indicators: Williams %R (14) ===
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14)
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # === 1w Indicators: EMA (50) ===
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to primary timeframe (12h)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state
    position = 0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        wr = williams_r_aligned[i]
        ema50 = ema_50_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # Get 12h volume average
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            if wr > -50:  # Exit when Williams %R crosses above -50
                exit_signal = True
        
        elif position == -1:  # Short position
            if wr < -50:  # Exit when Williams %R crosses below -50
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Williams %R < -80 (oversold) AND price > EMA50 (uptrend) AND volume > 1.5x 20-period avg
            if (wr < -80) and (price > ema50) and (vol > 1.5 * vol_ma_20[i]):
                signals[i] = 0.25
                position = 1
            
            # SHORT: Williams %R > -20 (overbought) AND price < EMA50 (downtrend) AND volume > 1.5x 20-period avg
            elif (wr > -20) and (price < ema50) and (vol > 1.5 * vol_ma_20[i]):
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "12h_1dWilliamsR_Extreme_1wEMA50_VolumeConfirmation_V1"
timeframe = "12h"
leverage = 1.0