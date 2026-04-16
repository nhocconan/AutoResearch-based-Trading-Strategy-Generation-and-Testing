#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and ADX(14) trend filter.
# Long when price breaks above 20-period high AND 1d volume > 1.5x 20-period average AND 1d ADX > 25 (trending).
# Short when price breaks below 20-period low AND 1d volume > 1.5x 20-period average AND 1d ADX > 25.
# Uses discrete position size 0.30. Donchian captures breakouts, volume confirms participation, ADX filters chop.
# Designed to work in bull (breakout longs) and bear (breakout shorts) markets.
# Target: 100-200 trades over 4 years (25-50/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    open_time = prices['open_time']
    
    # === 4h Indicators: Donchian(20) ===
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = high_ma.values
    donchian_low = low_ma.values
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    atr_values = atr.values
    
    # Get 1d data once before loop for volume and ADX filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === 1d Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (1.5 * vol_ma_1d)
    
    # === 1d Indicators: ADX(14) for trend filter ===
    # True Range
    tr1_1d = pd.Series(high_1d).diff()
    tr2_1d = pd.Series(low_1d).diff().abs()
    tr3_1d = pd.Series(close_1d).shift(1).diff().abs()
    tr_1d = pd.concat([tr1_1d, tr2_1d, tr3_1d], axis=1).max(axis=1)
    atr_1d = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # Directional Movement
    up_move_1d = pd.Series(high_1d).diff()
    down_move_1d = -pd.Series(low_1d).diff()
    plus_dm_1d = np.where((up_move_1d > down_move_1d) & (up_move_1d > 0), up_move_1d, 0)
    minus_dm_1d = np.where((down_move_1d > up_move_1d) & (down_move_1d > 0), down_move_1d, 0)
    
    # Smoothed DM
    plus_dm_smooth_1d = pd.Series(plus_dm_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    minus_dm_smooth_1d = pd.Series(minus_dm_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # Directional Indicators
    plus_di_1d = 100 * plus_dm_smooth_1d / atr_1d
    minus_di_1d = 100 * minus_dm_smooth_1d / atr_1d
    
    # DX and ADX
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = pd.Series(dx_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    adx_1d_values = adx_1d.values
    
    # Align 1d indicators to 4h timeframe
    volume_spike_aligned = align_htf_to_ltf(open_time, df_1d, volume_spike_1d)
    adx_aligned = align_htf_to_ltf(open_time, df_1d, adx_1d_values)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 34 periods needed for ADX, 20 for Donchian/volume MA)
    warmup = 40
    
    # Track position state and entry price for ATR-based stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(atr_values[i]) or np.isnan(volume_spike_aligned[i]) or
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        atr_val = atr_values[i]
        vol_spike = volume_spike_aligned[i]
        adx_val = adx_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Stoploss: price < entry_price - 2.0 * ATR
            if price < entry_price - 2.0 * atr_val:
                exit_signal = True
            # Exit if ADX drops below 20 (trend weakening)
            elif adx_val < 20:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Stoploss: price > entry_price + 2.0 * ATR
            if price > entry_price + 2.0 * atr_val:
                exit_signal = True
            # Exit if ADX drops below 20 (trend weakening)
            elif adx_val < 20:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: price > Donchian high AND 1d volume spike AND 1d ADX > 25 (trending)
            if price > donchian_high[i] and vol_spike and adx_val > 25:
                signals[i] = 0.30
                position = 1
                entry_price = price
            
            # SHORT: price < Donchian low AND 1d volume spike AND 1d ADX > 25 (trending)
            elif price < donchian_low[i] and vol_spike and adx_val > 25:
                signals[i] = -0.30
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.30
    
    return signals

name = "4h_Donchian20_1dVolumeSpike_ADXFilter_V1"
timeframe = "4h"
leverage = 1.0