#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA50 filter and volume confirmation.
# Bull Power = High - EMA(close,50); Bear Power = Low - EMA(close,50)
# Long when Bull Power > 0 AND Bear Power < 0 AND 1d EMA50 rising AND 6h volume > 1.5x 20-period average
# Short when Bull Power < 0 AND Bear Power > 0 AND 1d EMA50 falling AND 6h volume > 1.5x 20-period average
# Exit when Elder Ray signals weaken (Bull Power < 0 for long, Bear Power > 0 for short)
# Uses 1d EMA50 for trend filter (more reliable than ADX in ranging markets) and volume spike for momentum confirmation.
# Target: 60-120 total trades over 4 years (15-30/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate 1d EMA50
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if np.isnan(ema_50_1d_aligned[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        ema_50_val = ema_50_1d_aligned[i]
        vol = volume[i]
        
        # === 6h Indicators: Elder Ray (Bull/Bear Power) ===
        # Need 50-period EMA of 6h close for Elder Ray calculation
        if i >= 50:
            ema_50_6h = np.mean(close[i-49:i+1])  # Simple average for efficiency, can be EMA
            bull_power = high[i] - ema_50_6h
            bear_power = low[i] - ema_50_6h
        else:
            # Not enough data for Elder Ray, stay flat
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume filter: volume > 1.5x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
            vol_filter = vol > 1.5 * vol_ma_20 if vol_ma_20 > 0 else False
        else:
            vol_filter = False
        
        # Trend filter: 1d EMA50 direction (rising/falling)
        if i >= 1:
            ema_50_prev = ema_50_1d_aligned[i-1]
            ema_50_rising = ema_50_val > ema_50_prev
            ema_50_falling = ema_50_val < ema_50_prev
        else:
            ema_50_rising = False
            ema_50_falling = False
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Bull Power becomes negative (weakening bullish momentum)
            if bull_power < 0:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Bear Power becomes positive (weakening bearish momentum)
            if bear_power > 0:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Bull Power > 0 AND Bear Power < 0 (strong bullish momentum)
            #        AND 1d EMA50 rising (uptrend) AND volume spike
            if bull_power > 0 and bear_power < 0 and ema_50_rising and vol_filter:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Bull Power < 0 AND Bear Power > 0 (strong bearish momentum)
            #        AND 1d EMA50 falling (downtrend) AND volume spike
            elif bull_power < 0 and bear_power > 0 and ema_50_falling and vol_filter:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_ElderRay_EMA50_VolumeFilter_V2"
timeframe = "6h"
leverage = 1.0