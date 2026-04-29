#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d EMA34 trend filter + volume confirmation
# Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trend direction and strength.
# In bull markets: Lips > Teeth > Jaw (all aligned up) = strong uptrend.
# In bear markets: Lips < Teeth < Jaw (all aligned down) = strong downtrend.
# 1d EMA34 filters for higher timeframe trend alignment to avoid counter-trend trades.
# Volume confirmation ensures breakouts have participation.
# Target: 12-30 trades/year (50-120 total over 4 years) to minimize fee drag.

name = "12h_WilliamsAlligator_1dEMA34_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Load HTF data ONCE before loop for 1d calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams Alligator on 12h timeframe
    # Jaw: 13-period SMMA (smoothed moving average) of median price, shifted 8 bars
    # Teeth: 8-period SMMA of median price, shifted 5 bars
    # Lips: 5-period SMMA of median price, shifted 3 bars
    median_price = (high + low) / 2.0
    
    # SMMA calculation: first value is SMA, subsequent values are smoothed
    def smma(values, period):
        result = np.full_like(values, np.nan)
        if len(values) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(values[:period])
        # Subsequent values: (prev * (period-1) + current) / period
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    smma_jaw = smma(median_price, 13)
    smma_teeth = smma(median_price, 8)
    smma_lips = smma(median_price, 5)
    
    # Apply shifts: Jaw shifted 8, Teeth shifted 5, Lips shifted 3
    jaw = np.roll(smma_jaw, 8)
    teeth = np.roll(smma_teeth, 5)
    lips = np.roll(smma_lips, 3)
    
    # Set NaN for shifted positions
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 13+8)  # warmup for EMA34, volume, Alligator shifts
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_jaw = jaw[i]
        curr_teeth = teeth[i]
        curr_lips = lips[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Alligator alignment conditions
        bullish_alignment = (curr_lips > curr_teeth) and (curr_teeth > curr_jaw)
        bearish_alignment = (curr_lips < curr_teeth) and (curr_teeth < curr_jaw)
        
        # Handle position exits
        if position == 1:  # Long position
            # Exit conditions:
            # 1. Alligator loses bullish alignment (teeth crosses below jaw or lips crosses below teeth)
            # 2. Price crosses below 1d EMA34 (trend change)
            if (not bullish_alignment or curr_close < curr_ema_34_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions:
            # 1. Alligator loses bearish alignment (teeth crosses above jaw or lips crosses above teeth)
            # 2. Price crosses above 1d EMA34 (trend change)
            if (not bearish_alignment or curr_close > curr_ema_34_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: bullish Alligator alignment + price above 1d EMA34 + volume confirm
            if (bullish_alignment and
                curr_close > curr_ema_34_1d and
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short entry: bearish Alligator alignment + price below 1d EMA34 + volume confirm
            elif (bearish_alignment and
                  curr_close < curr_ema_34_1d and
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals