#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h breakout strategy using 4h Donchian channels for direction and 1d volatility regime filter
# 4h Donchian(20) provides adaptive support/resistance based on recent volatility
# 1d ATR ratio (ATR10/ATR30) identifies high volatility regimes where breakouts are more reliable
# Volume confirmation (>1.3x 20-period average) ensures participation
# Session filter (08-20 UTC) reduces noise during low-liquidity hours
# Trailing stop (2.0x ATR) manages risk while allowing trends to develop
# Position size: 0.20 (20% of capital) to control drawdown
# Target: 60-120 total trades over 4 years (15-30/year) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h data (for Donchian channels) ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # === 1d data (for volatility regime filter) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 4h Donchian Channel (20) ===
    high_roll_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_roll_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_upper = align_htf_to_ltf(prices, df_4h, high_roll_4h)
    donchian_lower = align_htf_to_ltf(prices, df_4h, low_roll_4h)
    
    # === 1d ATR Ratio (ATR10/ATR30) for volatility regime ===
    # Calculate ATR for 1d timeframe
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_10_1d = pd.Series(tr_1d).ewm(span=10, adjust=False, min_periods=10).mean().values
    atr_30_1d = pd.Series(tr_1d).ewm(span=30, adjust=False, min_periods=30).mean().values
    atr_ratio_1d = atr_10_1d / (atr_30_1d + 1e-10)  # Avoid division by zero
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    # === 4h Volume Confirmation (20-period average) ===
    vol_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    
    # === 4h ATR (10) for trailing stop ===
    tr1_4h = high_4h - low_4h
    tr2_4h = np.abs(high_4h - np.roll(close_4h, 1))
    tr3_4h = np.abs(low_4h - np.roll(close_4h, 1))
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    tr_4h[0] = high_4h[0] - low_4h[0]
    atr_4h = pd.Series(tr_4h).ewm(span=10, adjust=False, min_periods=10).mean().values
    atr_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # Already datetime64[ms], .hour works directly
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0  # For long trailing stop
    lowest_since_entry = 0.0   # For short trailing stop
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or
            np.isnan(atr_ratio_aligned[i]) or
            np.isnan(vol_ma_aligned[i]) or
            np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        if not in_session:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            highest_since_entry = 0.0
            lowest_since_entry = 0.0
            continue
        
        price = close[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        vol_confirm = volume[i] > vol_ma_aligned[i] * 1.3  # 1.3x average volume for confirmation
        atr_val = atr_aligned[i]
        vol_regime = atr_ratio_aligned[i] > 0.8  # High volatility regime (ATR10 > 0.8 * ATR30)
        
        # === TRAILING STOP LOGIC ===
        if position == 1:  # Long position
            # Update highest price since entry
            if price > highest_since_entry:
                highest_since_entry = price
            # Trail stop: exit if price drops 2.0*ATR from high
            if price < highest_since_entry - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                continue
        
        elif position == -1:  # Short position
            # Update lowest price since entry
            if price < lowest_since_entry:
                lowest_since_entry = price
            # Trail stop: exit if price rises 2.0*ATR from low
            if price > lowest_since_entry + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                lowest_since_entry = 0.0
                continue
        
        # === EXIT LOGIC (Donchian reversal) ===
        if position == 1:  # Long position
            # Exit when price breaks below lower Donchian channel
            if price < lower:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when price breaks above upper Donchian channel
            if price > upper:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                lowest_since_entry = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0 and in_session:
            # Require volume confirmation and high volatility regime
            if vol_confirm and vol_regime:
                # Long when price breaks above upper channel
                if price > upper:
                    signals[i] = 0.20
                    position = 1
                    entry_price = price
                    highest_since_entry = price
                    continue
                # Short when price breaks below lower channel
                elif price < lower:
                    signals[i] = -0.20
                    position = -1
                    entry_price = price
                    lowest_since_entry = price
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.20
        elif position == -1:
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals

name = "1h_Donchian20_1dATRRatio_VolumeConfirm_SessionFilter"
timeframe = "1h"
leverage = 1.0