#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elliott Wave Structure + 1d Volume Profile Confirmation
# Uses the principle that major trends unfold in 5-wave impulses followed by 3-wave corrections.
# Identifies wave 3 (strongest impulse) and wave C (strong correction) using:
# 1) 6h price structure: Higher highs/lows in uptrend, lower highs/lows in downtrend
# 2) 1d volume confirmation: Volume must expand in direction of trend
# 3) Entry on pullbacks to 6h EMA(21) during established trends
# Works in bull markets (catch wave 3) and bear markets (catch wave C).
# Target: 50-150 total trades over 4 years (12-37/year) with disciplined entries.

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
    
    # === 1d data (higher timeframe for volume confirmation) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === 6h EMA(21) for dynamic support/resistance ===
    ema_21_6h = pd.Series(close_6h).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # === 6h Higher Highs/Lower Lows Structure ===
    # Higher High: current high > previous high AND previous high > high before that
    hh_condition = (high_6h[2:] > high_6h[1:-1]) & (high_6h[1:-1] > high_6h[:-2])
    hl_condition = (low_6h[2:] > low_6h[1:-1]) & (low_6h[1:-1] > low_6h[:-2])
    # Lower High: current high < previous high AND previous high < high before that
    lh_condition = (high_6h[2:] < high_6h[1:-1]) & (high_6h[1:-1] < high_6h[:-2])
    ll_condition = (low_6h[2:] < low_6h[1:-1]) & (low_6h[1:-1] < low_6h[:-2])
    
    # Pad to original length
    hh_hl = np.zeros(len(high_6h), dtype=bool)
    lh_ll = np.zeros(len(high_6h), dtype=bool)
    hh_hl[2:] = hh_condition & hl_condition  # Higher High AND Higher Low
    lh_ll[2:] = lh_condition & ll_condition  # Lower High AND Lower Low
    
    # === Trend Structure Signals ===
    # Uptrend structure: HH&HL
    # Downtrend structure: LH&LL
    struct_score = np.zeros(len(high_6h))
    struct_score[hh_hl] = 1   # Uptrend structure
    struct_score[lh_ll] = -1  # Downtrend structure
    
    # Smooth structure signal to avoid whipsaw
    struct_smoothed = pd.Series(struct_score).rolling(window=5, min_periods=1).sum().values
    # Normalize to [-1, 1]
    struct_smoothed = np.clip(struct_smoothed / 5.0, -1, 1)
    
    # === 1d Volume Confirmation ===
    # Volume must be above average to confirm trend strength
    vol_avg_1d = pd.Series(volume_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ratio_1d = volume_1d / vol_avg_1d
    # Volume confirmation: current volume > 1.2 * average
    vol_confirmed = vol_ratio_1d > 1.2
    
    # === Align 1d data to 6s timeframe ===
    struct_aligned = align_htf_to_ltf(prices, df_6h, struct_smoothed)
    vol_confirmed_aligned = align_htf_to_ltf(prices, df_1d, vol_confirmed.astype(float))
    
    # === 6d EMA alignment for entry timing ===
    ema_21_aligned = align_htf_to_ltf(prices, df_6h, ema_21_6h)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 60
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(struct_aligned[i]) or np.isnan(vol_confirmed_aligned[i]) or 
            np.isnan(ema_21_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        struct_val = struct_aligned[i]
        vol_conf = vol_confirmed_aligned[i] > 0.5  # Convert back to boolean
        ema_val = ema_21_aligned[i]
        
        # === STOPLOSS LOGICS ===
        if position == 1:  # Long position
            if price < ema_val - 0.02 * ema_val:  # 2% below EMA
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            if price > ema_val + 0.02 * ema_val:  # 2% above EMA
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when structure breaks down or volume fails
            if struct_val < 0.3 or not vol_conf:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when structure breaks down or volume fails
            if struct_val > -0.3 or not vol_conf:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Require clear structure and volume confirmation
            if abs(struct_val) > 0.5 and vol_conf:
                # Long when uptrend structure and price near EMA support
                if struct_val > 0.5 and price <= ema_val * 1.01:  # Within 1% above EMA
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    continue
                # Short when downtrend structure and price near EMA resistance
                elif struct_val < -0.5 and price >= ema_val * 0.99:  # Within 1% below EMA
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

name = "6h_ElliottWave_Structure_Volume_Confirm_v1"
timeframe = "6h"
leverage = 1.0