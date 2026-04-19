#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d ATR breakout with 1w trend filter (Supertrend) and volume confirmation.
# Enter long when price breaks above ATR-based upper band in uptrend (Supertrend green).
# Enter short when price breaks below ATR-based lower band in downtrend (Supertrend red).
# Volume confirmation ensures breakout validity. Designed for low frequency (target: 10-25 trades/year)
# to minimize fee drag. Works in bull/bear markets by following the weekly trend.
name = "1d_ATRBreakout_Supertrend_Weekly_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Supertrend trend filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Supertrend (10, 3.0) on weekly
    atr_period = 10
    atr_multiplier = 3.0
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR using Wilder's smoothing
    atr = np.zeros_like(close_1w)
    atr[atr_period] = np.mean(tr[1:atr_period+1])
    for i in range(atr_period+1, len(tr)):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Supertrend calculation
    hl2 = (high_1w + low_1w) / 2
    upper_band = hl2 + (atr_multiplier * atr)
    lower_band = hl2 - (atr_multiplier * atr)
    
    supertrend = np.zeros_like(close_1w)
    direction = np.ones_like(close_1w)  # 1 for uptrend, -1 for downtrend
    
    supertrend[atr_period] = upper_band[atr_period]
    direction[atr_period] = 1
    
    for i in range(atr_period+1, len(close_1w)):
        if close_1w[i] <= supertrend[i-1]:
            direction[i] = -1
        else:
            direction[i] = 1
        
        if direction[i] == 1 and direction[i-1] == -1:
            supertrend[i] = lower_band[i]
        elif direction[i] == -1 and direction[i-1] == 1:
            supertrend[i] = upper_band[i]
        elif direction[i] == 1:
            supertrend[i] = max(lower_band[i], supertrend[i-1])
        else:
            supertrend[i] = min(upper_band[i], supertrend[i-1])
    
    # Align Supertrend direction to daily
    supertrend_direction_aligned = align_htf_to_ltf(prices, df_1w, direction)
    
    # ATR for daily breakout bands
    atr_period_d = 14
    atr_multiplier_d = 2.0
    
    # True Range for daily
    tr1_d = high - low
    tr2_d = np.abs(high - np.roll(close, 1))
    tr3_d = np.abs(low - np.roll(close, 1))
    tr1_d[0] = 0
    tr2_d[0] = 0
    tr3_d[0] = 0
    tr_d = np.maximum(tr1_d, np.maximum(tr2_d, tr3_d))
    
    # ATR using Wilder's smoothing
    atr_d = np.zeros_like(close)
    atr_d[atr_period_d] = np.mean(tr_d[1:atr_period_d+1])
    for i in range(atr_period_d+1, len(tr_d)):
        atr_d[i] = (atr_d[i-1] * (atr_period_d-1) + tr_d[i]) / atr_period_d
    
    # Calculate daily breakout bands
    hl2_d = (high + low) / 2
    upper_band_d = hl2_d + (atr_multiplier_d * atr_d)
    lower_band_d = hl2_d - (atr_multiplier_d * atr_d)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(atr_period_d + 1, atr_period + 1, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(supertrend_direction_aligned[i]) or np.isnan(atr_d[i]) or 
            np.isnan(upper_band_d[i]) or np.isnan(lower_band_d[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        supertrend_dir = supertrend_direction_aligned[i]
        atr_val = atr_d[i]
        upper_band = upper_band_d[i]
        lower_band = lower_band_d[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 2.0 * vol_ma
        
        if position == 0:
            # Enter long if price breaks above upper band, uptrend, and volume confirmation
            if price > upper_band and supertrend_dir == 1 and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Enter short if price breaks below lower band, downtrend, and volume confirmation
            elif price < lower_band and supertrend_dir == -1 and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price closes below Supertrend (trend reversal)
            if supertrend_dir == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price closes above Supertrend (trend reversal)
            if supertrend_dir == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals