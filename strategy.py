#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA34 trend filter and volume confirmation
# Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# In bull markets (price > 1d EMA34): look for Bull Power expansion + volume spike for longs
# In bear markets (price < 1d EMA34): look for Bear Power expansion + volume spike for shorts
# Volume confirmation ensures institutional participation. Works in both regimes by adapting to trend.
# Target: 50-150 trades over 4 years (12-37/year) to minimize fee drag while maintaining statistical significance.

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
    
    # === 1d data (HTF for trend filter) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # === 1d EMA34 for trend filter ===
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === 6h EMA13 for Elder Ray calculation ===
    ema13_6h = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # === Elder Ray components ===
    bull_power = high_6h - ema13_6h  # Bull Power: High - EMA13
    bear_power = low_6h - ema13_6h   # Bear Power: Low - EMA13
    
    # === 6h Volume confirmation (20-period MA) ===
    vol_ma_20_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_6h > (2.0 * vol_ma_20_6h)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i]) or
            np.isnan(vol_spike[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        ema34 = ema34_1d_aligned[i]
        bp = bull_power[i]
        br = bear_power[i]
        vol_conf = vol_spike[i]
        
        # === STOPLOSS LOGIC (ATR-based) ===
        if position == 1:  # Long position
            atr_6h = np.abs(high_6h - low_6h)
            atr_ma = pd.Series(atr_6h).rolling(window=14, min_periods=14).mean().values
            atr_val = atr_ma[i]
            if price < entry_price - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            atr_6h = np.abs(high_6h - low_6h)
            atr_ma = pd.Series(atr_6h).rolling(window=14, min_periods=14).mean().values
            atr_val = atr_ma[i]
            if price > entry_price + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when Bull Power turns negative (momentum fading)
            if bp <= 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when Bear Power turns positive (momentum fading)
            if br >= 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Bull market: price > 1d EMA34 -> look for longs
            if price > ema34:
                # Long: Bull Power expanding (increasing) + volume spike
                if bp > 0 and bp > bull_power[i-1] and vol_conf:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    continue
            # Bear market: price < 1d EMA34 -> look for shorts
            elif price < ema34:
                # Short: Bear Power expanding (more negative) + volume spike
                if br < 0 and br < bear_power[i-1] and vol_conf:
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