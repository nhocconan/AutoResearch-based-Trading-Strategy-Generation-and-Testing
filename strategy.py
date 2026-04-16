#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d EMA34 Trend Filter + Volume Spike
# Elder Ray measures bull/bear power relative to EMA13. Strong bull power + price above 1d EMA34 = long.
# Strong bear power + price below 1d EMA34 = short. Volume spike (>2x 20-period) confirms conviction.
# ATR trailing stop (2.5x) manages risk. Works in bull/bear as EMA34 adapts and Elder Ray shows momentum exhaustion.
# Target: 50-150 total trades over 4 years (12-37/year). Discrete size 0.25 minimizes fee churn.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h data (primary timeframe) ===
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # === 1d data (HTF for EMA34 trend filter) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d EMA34 for trend filter ===
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === 6h EMA13 for Elder Ray calculation ===
    close_6h_series = pd.Series(close_6h)
    ema13_6h = close_6h_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_6h_aligned = align_htf_to_ltf(prices, df_6h, ema13_6h)
    
    # === Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 ===
    bull_power = high_6h - ema13_6h
    bear_power = low_6h - ema13_6h
    
    # Smooth Elder Ray with EMA(8) to reduce noise
    bull_power_series = pd.Series(bull_power)
    bear_power_series = pd.Series(bear_power)
    bull_power_smooth = bull_power_series.ewm(span=8, adjust=False, min_periods=8).mean().values
    bear_power_smooth = bear_power_series.ewm(span=8, adjust=False, min_periods=8).mean().values
    
    # Align smoothed Elder Ray to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_6h, bull_power_smooth)
    bear_power_aligned = align_htf_to_ltf(prices, df_6h, bear_power_smooth)
    
    # === 6h Volume Confirmation (20-period average) ===
    vol_ma_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_20)
    
    # === 6h ATR (15) for trailing stop ===
    atr_6h = np.maximum(high_6h - low_6h, np.maximum(np.abs(high_6h - np.roll(close_6h, 1)), np.abs(low_6h - np.roll(close_6h, 1))))
    atr_6h[0] = high_6h[0] - low_6h[0]  # Fix first value
    atr_ma = pd.Series(atr_6h).ewm(span=15, adjust=False, min_periods=15).mean().values
    atr_aligned = align_htf_to_ltf(prices, df_6h, atr_ma)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0  # For long trailing stop
    lowest_since_entry = 0.0   # For short trailing stop
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(bull_power_aligned[i]) or
            np.isnan(bear_power_aligned[i]) or
            np.isnan(vol_ma_aligned[i]) or
            np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        ema34 = ema34_1d_aligned[i]
        bull = bull_power_aligned[i]
        bear = bear_power_aligned[i]
        vol_confirm = volume[i] > vol_ma_aligned[i] * 2.0  # 2x average volume for confirmation
        atr_val = atr_aligned[i]
        
        # === TRAILING STOP LOGIC ===
        if position == 1:  # Long position
            # Update highest price since entry
            if price > highest_since_entry:
                highest_since_entry = price
            # Trail stop: exit if price drops 2.5*ATR from high
            if price < highest_since_entry - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                continue
        
        elif position == -1:  # Short position
            # Update lowest price since entry
            if price < lowest_since_entry:
                lowest_since_entry = price
            # Trail stop: exit if price rises 2.5*ATR from low
            if price > lowest_since_entry + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                lowest_since_entry = 0.0
                continue
        
        # === EXIT LOGIC (Elder Ray divergence) ===
        if position == 1:  # Long position
            # Exit when bull power turns negative (momentum loss)
            if bull < 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when bear power turns positive (momentum loss)
            if bear > 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                lowest_since_entry = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Require volume confirmation
            if vol_confirm:
                # Long when bull power > 0 AND price above 1d EMA34 (uptrend)
                if bull > 0 and price > ema34:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    highest_since_entry = price
                    continue
                # Short when bear power < 0 AND price below 1d EMA34 (downtrend)
                elif bear < 0 and price < ema34:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                    lowest_since_entry = price
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_ElderRay_EMA34Trend_VolumeSpike_ATRTrail"
timeframe = "6h"
leverage = 1.0