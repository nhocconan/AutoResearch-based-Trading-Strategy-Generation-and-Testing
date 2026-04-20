#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and ADX trend filter
# Works in bull markets (breakouts continue) and bear markets (breakdowns continue)
# Uses price channels as objective support/resistance, volume to confirm conviction,
# and ADX to avoid choppy markets. Target: 20-40 trades/year per symbol.

name = "4h_1d_Donchian20_Breakout_Volume_ADXFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === 1d Volume Confirmation (average volume) ===
    volume_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = volume_1d / np.where(vol_ma20_1d > 0, vol_ma20_1d, np.nan)
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # === 4h Donchian Channel (20) ===
    high = prices['high'].values
    low = prices['low'].values
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h ADX Filter ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), np.maximum(np.roll(low, 1) - low, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    tr_ma = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_dm_ma = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_ma = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * plus_dm_ma / np.where(tr_ma > 0, tr_ma, np.nan)
    minus_di = 100 * minus_dm_ma / np.where(tr_ma > 0, tr_ma, np.nan)
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) > 0, (plus_di + minus_di), np.nan)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Get values
        close_val = close[i]
        donch_high = donchian_high[i]
        donch_low = donchian_low[i]
        vol_ratio_val = vol_ratio_1d_aligned[i]
        adx_val = adx[i]
        
        # Skip if any value is NaN
        if (np.isnan(donch_high) or np.isnan(donch_low) or 
            np.isnan(vol_ratio_val) or np.isnan(adx_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Breakout with volume confirmation and ADX > 25 (trending)
            if close_val > donch_high and vol_ratio_val > 1.5 and adx_val > 25:
                # Break above Donchian high
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif close_val < donch_low and vol_ratio_val > 1.5 and adx_val > 25:
                # Break below Donchian low
                signals[i] = -0.25
                position = -1
                entry_price = close_val
        
        elif position == 1:
            # Long exit: stop loss or return to Donchian low
            atr_proxy = (high[i] - low[i])  # Simplified ATR proxy
            if close_val <= entry_price - 2.0 * atr_proxy:
                # Stop loss hit
                signals[i] = 0.0
                position = 0
            elif close_val < donch_low:
                # Return to Donchian low
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: stop loss or return to Donchian high
            atr_proxy = (high[i] - low[i])  # Simplified ATR proxy
            if close_val >= entry_price + 2.0 * atr_proxy:
                # Stop loss hit
                signals[i] = 0.0
                position = 0
            elif close_val > donch_high:
                # Return to Donchian high
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals