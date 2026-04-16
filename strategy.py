#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h data (primary) ===
    df_6h = get_htf_data(prices, '6h')
    close_6h = df_6h['close'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    volume_6h = df_6h['volume'].values
    
    # === 1d data (HTF for ATR and ATR ratio) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d ATR(7) and ATR(30) for volatility ratio ===
    def calculate_atr(high, low, close, period):
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        tr = np.concatenate([[np.nan], tr])
        
        atr = np.full_like(tr, np.nan)
        if len(tr) >= period:
            atr[period-1] = np.nanmean(tr[1:period])
            for i in range(period, len(tr)):
                atr[i] = atr[i-1] - (atr[i-1]/period) + tr[i]
        return atr
    
    atr_7 = calculate_atr(high_1d, low_1d, close_1d, 7)
    atr_30 = calculate_atr(high_1d, low_1d, close_1d, 30)
    atr_ratio = atr_7 / atr_30  # Volatility expansion signal
    
    # === 6h Donchian channel (20-period) ===
    def calculate_donchian(high, low, period):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(low, np.nan)
        for i in range(period-1, len(high)):
            upper[i] = np.max(high[i-(period-1):i+1])
            lower[i] = np.min(low[i-(period-1):i+1])
        return upper, lower
    
    donchian_upper, donchian_lower = calculate_donchian(high_6h, low_6h, 20)
    
    # === 6h volume ratio for confirmation ===
    vol_ma_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume_6h / vol_ma_20
    
    # Align all HTF data to 6h timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_6h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_6h, donchian_lower)
    vol_ratio_aligned = align_htf_to_ltf(prices, df_6h, vol_ratio)
    
    signals = np.zeros(n)
    
    # Warmup: enough for ATR and Donchian calculations
    warmup = 60
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(atr_ratio_aligned[i]) or 
            np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(vol_ratio_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        upper = donchian_upper_aligned[i]
        lower = donchian_lower_aligned[i]
        vol_ratio_val = vol_ratio_aligned[i]
        atr_ratio_val = atr_ratio_aligned[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit: price closes below lower band OR volatility contraction
            if price < lower or atr_ratio_val < 0.8:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit: price closes above upper band OR volatility contraction
            if price > upper or atr_ratio_val < 0.8:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volatility expansion breakout: ATR(7)/ATR(30) > 1.2 + volume confirmation
            if atr_ratio_val > 1.2 and vol_ratio_val > 1.3:
                # LONG: Break above upper band
                if price > upper:
                    signals[i] = 0.25
                    position = 1
                    continue
                # SHORT: Break below lower band
                elif price < lower:
                    signals[i] = -0.25
                    position = -1
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_ATR_Volatility_Breakout_Donchian"
timeframe = "6h"
leverage = 1.0