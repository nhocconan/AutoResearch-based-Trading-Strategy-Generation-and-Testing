#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + Elder Ray + Volume Confirmation
# Uses Alligator (Jaw/Teeth/Lips) to identify trend direction, Elder Ray (Bull/Bear Power) for momentum,
# and volume > 1.5x average for confirmation. Works in both bull and bear markets by following trend.
# Target: 50-150 total trades over 4 years (12-37/year) with disciplined entries.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h data (primary timeframe) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # === 1d data (higher timeframe for trend filter) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 12h Williams Alligator (13,8,5 SMAs) ===
    jaw = pd.Series(close_12h).rolling(window=13, min_periods=13).mean().values  # 13-period SMA
    teeth = pd.Series(close_12h).rolling(window=8, min_periods=8).mean().values   # 8-period SMA
    lips = pd.Series(close_12h).rolling(window=5, min_periods=5).mean().values    # 5-period SMA
    
    # Align Alligator components to lower timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # === 12h Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13) ===
    ema13 = pd.Series(close_12h).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_12h - ema13
    bear_power = low_12h - ema13
    
    # Align Elder Ray components
    bull_power_aligned = align_htf_to_ltf(prices, df_12h, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_12h, bear_power)
    
    # === 12h volume spike detection ===
    vol_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_spike_12h = volume_12h > (1.5 * vol_ma_20_12h)
    vol_spike_aligned = align_htf_to_ltf(prices, df_12h, vol_spike_12h.astype(float))
    
    # === 1d ADX(14) for trend strength filter ===
    # Calculate True Range
    tr1 = pd.Series(high_1d).diff()
    tr2 = abs(pd.Series(high_1d).diff())
    tr3 = abs(pd.Series(low_1d).diff())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate Directional Movement
    up_move = pd.Series(high_1d).diff()
    down_move = -pd.Series(low_1d).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth DM and TR
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr_1d
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr_1d
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(vol_spike_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        bull_power_val = bull_power_aligned[i]
        bear_power_val = bear_power_aligned[i]
        vol_spike_val = vol_spike_aligned[i] > 0.5
        adx_val = adx_aligned[i]
        
        # === STOPLOSS LOGIC ===
        if position == 1:  # Long position
            atr_12h = np.abs(high_12h - low_12h)
            atr_ma = pd.Series(atr_12h).rolling(window=14, min_periods=14).mean().values
            atr_aligned = align_htf_to_ltf(prices, df_12h, atr_ma)
            atr_val = atr_aligned[i]
            if price < entry_price - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            atr_12h = np.abs(high_12h - low_12h)
            atr_ma = pd.Series(atr_12h).rolling(window=14, min_periods=14).mean().values
            atr_aligned = align_htf_to_ltf(prices, df_12h, atr_ma)
            atr_val = atr_aligned[i]
            if price > entry_price + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when Alligator reverses (lips < teeth) or Elder Ray turns negative
            if lips_val < teeth_val or bull_power_val < 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when Alligator reverses (lips > teeth) or Elder Ray turns positive
            if lips_val > teeth_val or bear_power_val > 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Require strong trend (ADX > 25) and volume spike
            if adx_val > 25 and vol_spike_val:
                # Alligator alignment: Lips > Teeth > Jaw = Uptrend
                # Alligator alignment: Lips < Teeth < Jaw = Downtrend
                if lips_val > teeth_val > jaw_val and bull_power_val > 0:
                    # Go long in uptrend with positive Bull Power
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    continue
                elif lips_val < teeth_val < jaw_val and bear_power_val < 0:
                    # Go short in downtrend with negative Bear Power
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

name = "12h_Alligator_ElderRay_Volume_AdxFilter_v1"
timeframe = "12h"
leverage = 1.0