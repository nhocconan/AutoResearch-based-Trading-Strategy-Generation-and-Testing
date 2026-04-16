#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w ATR expansion filter and volume confirmation.
# Long when price breaks above 20-period 1d high AND 1w ATR > 1.5x 20-period 1w ATR average AND volume > 1.2x 20-period 1d average volume.
# Short when price breaks below 20-period 1d low AND 1w ATR > 1.5x 20-period 1w ATR average AND volume > 1.2x 20-period 1d average volume.
# Exit when price crosses the 1d midpoint or ATR-based stoploss (2*ATR from entry).
# Uses discrete position size 0.25. Designed to capture volatility-expansion breakouts in both bull and bear markets.
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag while maintaining edge.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Indicators: Donchian Channel (20-period) and Volume MA ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    vol_1d = df_1d['volume'].values
    
    # Donchian upper and lower bands (20-period)
    dc_upper_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    dc_lower_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    dc_mid_1d = (dc_upper_1d + dc_lower_1d) / 2
    
    # 1d volume MA (20-period)
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # === 1w Indicators: ATR Expansion Filter ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = pd.Series(high_1w).diff()
    tr2 = pd.Series(low_1w).diff().abs()
    tr3 = pd.Series(close_1w).shift(1).diff().abs()
    tr_1w = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1w = pd.Series(tr_1w).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # 20-period ATR average
    atr_ma_1w = pd.Series(atr_1w).rolling(window=20, min_periods=20).mean().values
    atr_expansion = atr_1w > (1.5 * atr_ma_1w)
    
    # Align 1w indicators to 1d timeframe
    atr_expansion_aligned = align_htf_to_ltf(prices, df_1w, atr_expansion)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed)
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Calculate 1d ATR for stoploss
    tr1_1d = pd.Series(high_1d).diff()
    tr2_1d = pd.Series(low_1d).diff().abs()
    tr3_1d = pd.Series(close_1d).shift(1).diff().abs()
    tr_1d = pd.concat([tr1_1d, tr2_1d, tr3_1d], axis=1).max(axis=1)
    atr_1d_raw = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_1d_aligned = atr_1d_raw  # Already aligned as primary timeframe
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(dc_upper_1d[i]) or np.isnan(dc_lower_1d[i]) or np.isnan(dc_mid_1d[i]) or
            np.isnan(vol_ma_1d[i]) or np.isnan(atr_expansion_aligned[i]) or np.isnan(atr_1d_aligned[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_ma = vol_ma_1d[i]
        atr_exp = atr_expansion_aligned[i]
        atr_val = atr_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.2x 20-period average
        volume_confirm = volume[i] > (1.2 * vol_ma)
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price crosses below midpoint
            if price < dc_mid_1d[i]:
                exit_signal = True
            # ATR-based stoploss: 2*ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price crosses above midpoint
            if price > dc_mid_1d[i]:
                exit_signal = True
            # ATR-based stoploss: 2*ATR above entry
            elif price > entry_price + 2.0 * atr_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Donchian upper AND ATR expansion AND volume confirmation
            if price > dc_upper_1d[i] and atr_exp and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below Donchian lower AND ATR expansion AND volume confirmation
            elif price < dc_lower_1d[i] and atr_exp and volume_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "1d_Donchian20_1wATRExpansion_VolumeConfirm_V1"
timeframe = "1d"
leverage = 1.0