#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1w EMA50 trend filter + volume spike confirmation.
# Uses 12h timeframe for optimal trade frequency (target: 50-150 total trades over 4 years).
# Williams Alligator (JAW=13, TEETH=8, LIPS=5) provides trend direction and entry signals.
# 1w EMA50 ensures alignment with primary trend. Volume spike confirms momentum.
# ATR-based trailing stop for risk management. Discrete sizing 0.25 to minimize fee drag.
# Designed to work in both bull and bear markets by following the 1w trend.

name = "12h_WilliamsAlligator_1wEMA50_VolumeSpike"
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
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1w EMA50 trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1w volume regime (high volume when current volume > 1.5x 20-period MA)
    vol_1w = df_1w['volume'].values
    vol_ma_1w = pd.Series(vol_1w).rolling(window=20, min_periods=20).mean().values
    vol_regime_1w = vol_1w > (1.5 * vol_ma_1w)  # High volume regime
    vol_regime_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_regime_1w)
    
    # Calculate Williams Alligator on 12h data
    # JAW (Blue): 13-period SMMA, shifted 8 bars
    # TEETH (Red): 8-period SMMA, shifted 5 bars
    # LIPS (Green): 5-period SMMA, shifted 3 bars
    def smma(values, window):
        if len(values) < window:
            return np.full_like(values, np.nan)
        result = np.full_like(values, np.nan)
        # First value is SMA
        result[window-1] = np.mean(values[:window])
        # Subsequent values: SMMA = (PREV_SMMA * (N-1) + CLOSE) / N
        for i in range(window, len(values)):
            result[i] = (result[i-1] * (window-1) + values[i]) / window
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Apply shifts (Alligator lines are shifted into the future)
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    
    # Calculate ATR(14) for 12h data (for stoploss)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0
    lowest_low_since_entry = 0
    
    for i in range(100, n):
        # Get current values
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        ema_trend = ema_50_1w_aligned[i]
        vol_reg = vol_regime_1w_aligned[i]
        atr_val = atr[i]
        
        # Skip if any value is NaN
        if np.isnan(jaw_val) or np.isnan(teeth_val) or np.isnan(lips_val) or np.isnan(ema_trend) or np.isnan(vol_reg) or np.isnan(atr_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Volume confirmation: current 12h volume > 1.5x 20-period MA
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values[i]
        volume_spike = volume[i] > (1.5 * vol_ma_20)
        
        # Alligator conditions
        # Bullish alignment: Lips > Teeth > Jaw (green > red > blue)
        bullish_align = lips_val > teeth_val > jaw_val
        # Bearish alignment: Jaw > Teeth > Lips (blue > red > green)
        bearish_align = jaw_val > teeth_val > lips_val
        
        # Entry conditions
        # Long: Bullish Alligator alignment + price above Lips + volume spike + above 1w EMA50
        long_entry = bullish_align and (close[i] > lips_val) and volume_spike and (close[i] > ema_trend)
        # Short: Bearish Alligator alignment + price below Jaws + volume spike + below 1w EMA50
        short_entry = bearish_align and (close[i] < jaw_val) and volume_spike and (close[i] < ema_trend)
        
        # Exit conditions (ATR-based trailing stop)
        long_exit = False
        short_exit = False
        
        if position == 1:  # Long position
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            long_exit = close[i] < (highest_high_since_entry - 2.5 * atr_val)
        elif position == -1:  # Short position
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            short_exit = close[i] > (lowest_low_since_entry + 2.5 * atr_val)
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = high[i]
            elif short_entry:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = low[i]
        elif position == 1:
            if long_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            if short_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals