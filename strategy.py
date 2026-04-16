#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Williams %R (14) combined with 6h volume spike and price position relative to 6h EMA(50).
# Long when Williams %R < -80 (oversold) AND 6h volume > 1.5x 20-period median volume AND price > 6h EMA(50).
# Short when Williams %R > -20 (overbought) AND 6h volume > 1.5x 20-period median volume AND price < 6h EMA(50).
# Exits when Williams %R crosses above -50 (for longs) or below -50 (for shorts) OR ATR(14) stoploss hit (2.0x ATR).
# Williams %R identifies overextended moves prone to reversal. Volume confirmation ensures participation.
# 6h EMA(50) filter ensures we trade in direction of intermediate trend. Targets 12-37 trades/year to minimize fee drag.
# Works in bull markets (buy oversold dips in uptrend) and bear markets (sell overbought rallies in downtrend).

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data once before loop for Williams %R
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # === 12h Indicators: Williams %R (14-period) ===
    highest_high_14 = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_12h) / (highest_high_14 - lowest_low_14)
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # === 6h Indicators: EMA(50) for trend filter ===
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # === 6h Indicators: Volume Median (20-period) ===
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # === 6h Indicators: ATR (14-period) for stoploss ===
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_14 = pd.Series(true_range).rolling(window=14, min_periods=14).mean().values
    
    # Align all indicators to primary timeframe (6h)
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    # EMA, volume median, and ATR are already on primary timeframe
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(50, 20, 14)  # EMA50, volume median, ATR
    
    # Track position state and entry price for ATR stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50[i]) or
            np.isnan(vol_median_20[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values
        price = close[i]
        wr = williams_r_aligned[i]
        ema = ema_50[i]
        vol_median = vol_median_20[i]
        atr = atr_14[i]
        
        # Volume spike filter: current 6h volume > 1.5x median volume
        volume_spike = volume[i] > (vol_median * 1.5)
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when Williams %R crosses above -50 (momentum fading) OR ATR stoploss hit
            if wr > -50 or price <= entry_price - 2.0 * atr:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when Williams %R crosses below -50 (momentum fading) OR ATR stoploss hit
            if wr < -50 or price >= entry_price + 2.0 * atr:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Williams %R oversold (< -80) AND volume spike AND price above EMA(50) (uptrend filter)
            if wr < -80 and volume_spike and price > ema:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Williams %R overbought (> -20) AND volume spike AND price below EMA(50) (downtrend filter)
            elif wr > -20 and volume_spike and price < ema:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "6h_WilliamsR_VolumeSpike_EMA50Trend_ATRTrail2.0_v1"
timeframe = "6h"
leverage = 1.0