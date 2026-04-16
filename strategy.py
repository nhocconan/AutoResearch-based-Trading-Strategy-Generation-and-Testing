#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Bull/Bear Power with 1d ADX regime filter and 1w volume spike confirmation.
# Bull Power = High - EMA13, Bear Power = EMA13 - Low.
# Long when Bull Power > 0 AND Bear Power < 0 AND 1d ADX > 25 (trending) AND 1w volume > 1.5x 20-period average.
# Short when Bull Power < 0 AND Bear Power > 0 AND 1d ADX > 25 AND 1w volume > 1.5x 20-period average.
# Exit when Elder Power divergence occurs (Bull Power <= 0 for long, Bear Power <= 0 for short) or 6h ATR stop (2.5x).
# Uses discrete position size 0.28. Designed to capture institutional participation in strong trends across bull/bear markets.
# Target: 80-180 total trades over 4 years (20-45/year) balancing edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: Elder Ray Power (EMA13-based) ===
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # === 1d Indicators: ADX > 25 (strong trend filter) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d).diff()
    tr2 = pd.Series(low_1d).diff().abs()
    tr3 = pd.Series(close_1d).shift(1).diff().abs()
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    dm_plus = pd.Series(high_1d).diff()
    dm_minus = pd.Series(low_1d).diff().abs()
    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
    
    # Smoothed DM and TR
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_smooth = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * (dm_plus_smooth / atr_smooth)
    di_minus = 100 * (dm_minus_smooth / atr_smooth)
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    strong_trend = adx_aligned > 25
    
    # === 1w Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    df_1w = get_htf_data(prices, '1w')
    vol_1w = df_1w['volume'].values
    vol_ma_1w = pd.Series(vol_1w).rolling(window=20, min_periods=20).mean().values
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    volume_spike = volume > (1.5 * vol_ma_1w_aligned)
    
    # === 6h ATR for stoploss ===
    tr1_6h = pd.Series(high).diff()
    tr2_6h = pd.Series(low).diff().abs()
    tr3_6h = pd.Series(close).shift(1).diff().abs()
    tr_6h = pd.concat([tr1_6h, tr2_6h, tr3_6h], axis=1).max(axis=1)
    atr_6h_raw = pd.Series(tr_6h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_6h_aligned = atr_6h_raw  # Already aligned as primary timeframe
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for ADX/ATR/EMA)
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(strong_trend[i]) or
            np.isnan(volume_spike[i]) or np.isnan(atr_6h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        is_strong_trend = strong_trend[i]
        vol_spike = volume_spike[i]
        atr_val = atr_6h_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Bull Power turns non-positive (loss of buying pressure)
            if bull_val <= 0:
                exit_signal = True
            # ATR-based stoploss: 2.5*ATR below entry
            elif price < entry_price - 2.5 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Bear Power turns non-positive (loss of selling pressure)
            if bear_val <= 0:
                exit_signal = True
            # ATR-based stoploss: 2.5*ATR above entry
            elif price > entry_price + 2.5 * atr_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Bull Power > 0 AND Bear Power < 0 AND strong trend AND volume spike
            if bull_val > 0 and bear_val < 0 and is_strong_trend and vol_spike:
                signals[i] = 0.28
                position = 1
                entry_price = price
            
            # SHORT: Bull Power < 0 AND Bear Power > 0 AND strong trend AND volume spike
            elif bull_val < 0 and bear_val > 0 and is_strong_trend and vol_spike:
                signals[i] = -0.28
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.28
    
    return signals

name = "6h_ElderRay_BullBearPower_1dADX_1wVolumeSpike_V1"
timeframe = "6h"
leverage = 1.0