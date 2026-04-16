#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 12h EMA34 trend filter with volume spike confirmation
# Uses 6h primary timeframe with 12h HTF for trend direction and 1d HTF for volume confirmation.
# Elder Ray measures bull/bear power via EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13.
# Long when Bull Power > 0 and Bear Power < 0 (both bullish) with 12h EMA34 uptrend and 1d volume spike.
# Short when Bull Power < 0 and Bear Power > 0 (both bearish) with 12h EMA34 downtrend and 1d volume spike.
# Volume confirmation: 1d volume > 2.0x 20-period average.
# Target: 50-150 total trades over 4 years (12-37/year) to balance statistical significance and fee drag.
# Works in bull markets via long signals and in bear markets via short signals during strong trends.

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
    
    # === 12h data (HTF for EMA34 trend) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # === 1d data (HTF for volume confirmation) ===
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # === 6h EMA13 for Elder Ray ===
    ema13_6h = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_6h - ema13_6h
    bear_power = low_6h - ema13_6h
    
    # === 12h EMA34 trend filter ===
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # === 1d Volume confirmation ===
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (2.0 * vol_ma_20_1d)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or
            np.isnan(ema34_12h_aligned[i]) or
            np.isnan(vol_spike_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        bull = bull_power[i]
        bear = bear_power[i]
        trend = ema34_12h_aligned[i]
        vol_conf = vol_spike_aligned[i]
        
        # === STOPLOSS LOGIC (fixed 6%) ===
        if position == 1:  # Long position
            if price < entry_price * 0.94:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            if price > entry_price * 1.06:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === EXIT LOGIC (Elder Ray divergence) ===
        if position == 1:  # Long position
            # Exit when bull power turns negative or bear power turns positive
            if bull <= 0 or bear >= 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when bull power turns positive or bear power turns negative
            if bull >= 0 or bear <= 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Require volume confirmation
            if vol_conf:
                # Go long when both bull and bear power are bullish with uptrend
                if bull > 0 and bear < 0 and close_6h[i] > trend:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    continue
                # Go short when both bull and bear power are bearish with downtrend
                elif bull < 0 and bear > 0 and close_6h[i] < trend:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_ElderRay_EMA34Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0