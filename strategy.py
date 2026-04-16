#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Williams %R mean reversion with volume confirmation and ATR stoploss.
# Long when 1w Williams %R < -80 (oversold) + price > 1d EMA50 + 1d volume > 1.5x 20-period median volume.
# Short when 1w Williams %R > -20 (overbought) + price < 1d EMA50 + 1d volume > 1.5x 20-period median volume.
# Exit on opposite Williams %R level (-50) or when ATR-based trailing stop is hit (2.0 * ATR).
# Uses discrete position size 0.25. Williams %R identifies extremes in weekly momentum.
# EMA50 filter ensures alignment with weekly trend. Volume spike confirms participation.
# ATR stoploss manages risk. 1d timeframe targets 30-100 trades over 4 years (7-25/year) to minimize fee drag.
# Works in bull markets (mean reversion from oversold) and bear markets (mean reversion from overbought).

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data once before loop for Williams %R
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:  # Williams %R needs 14 periods, plus buffer
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # === 1w Indicators: Williams %R (14-period) ===
    highest_high_14 = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_1w) / (highest_high_14 - lowest_low_14)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # Get 1d data once before loop for EMA50, volume median, and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # EMA50 needs 50 periods
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    vol_1d = df_1d['volume'].values
    
    # === 1d Indicators: EMA50 ===
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === 1d Indicators: Volume Median (20-period) ===
    vol_median_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).median().values
    
    # === 1d Indicators: ATR (14-period) for stoploss ===
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = 0  # first period has no previous close
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align all indicators to primary timeframe (1d)
    williams_r_aligned = align_htf_to_ltf(prices, df_1w, williams_r)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    vol_median_aligned = align_htf_to_ltf(prices, df_1d, vol_median_20)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)  # for volume spike
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 50  # EMA50 needs 50 periods
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_median_aligned[i]) or np.isnan(atr_aligned[i]) or 
            np.isnan(vol_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values (aligned)
        price = close[i]
        wr = williams_r_aligned[i]
        ema50 = ema_50_aligned[i]
        vol_median = vol_median_aligned[i]
        atr = atr_aligned[i]
        current_vol_1d = vol_1d_aligned[i]
        
        # Volume spike filter: current 1d volume > 1.5x median volume
        volume_spike = current_vol_1d > (vol_median * 1.5)
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when Williams %R crosses above -50 (mean reversion complete) OR ATR stoploss hit
            if (wr >= -50) or (price <= entry_price - 2.0 * atr):
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when Williams %R crosses below -50 (mean reversion complete) OR ATR stoploss hit
            if (wr <= -50) or (price >= entry_price + 2.0 * atr):
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Williams %R < -80 (oversold) + price > EMA50 + volume spike
            if (wr < -80) and (price > ema50) and volume_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Williams %R > -20 (overbought) + price < EMA50 + volume spike
            elif (wr > -20) and (price < ema50) and volume_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "1d_1wWilliamsR_MeanReversion_VolumeSpike1.5x_EMA50Filter_ATRStop2.0_V1"
timeframe = "1d"
leverage = 1.0