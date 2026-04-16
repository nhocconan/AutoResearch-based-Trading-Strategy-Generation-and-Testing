#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR-based volatility filter and volume spike.
# Long when price breaks above 20-period 4h Donchian high AND volume > 2.0x 20-period 1d average AND 1d ATR ratio (ATR(14)/ATR(50)) > 1.2 (elevated volatility regime).
# Short when price breaks below 20-period 4h Donchian low AND volume > 2.0x 20-period 1d average AND 1d ATR ratio > 1.2.
# Exit when price crosses the 4h Donchian midpoint or ATR-based stoploss (2*ATR from entry).
# Uses discrete position size 0.25. Designed to capture volatility-expansion breakouts.
# Target: 50-150 total trades over 4 years (12-38/year) to balance edge and fee drag.
# Works in both bull and bear markets by requiring volume confirmation and volatility expansion.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: Donchian Channel (20-period) ===
    high_4h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_4h = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_high = high_4h
    donchian_low = low_4h
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # === 1d Indicators: Volume Spike (volume > 2.0x 20-period average) ===
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (2.0 * vol_ma_1d_aligned)
    
    # === 1d Indicators: ATR Ratio (ATR(14)/ATR(50) > 1.2) for volatility regime filter ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d).diff()
    tr2 = pd.Series(low_1d).diff().abs()
    tr3 = pd.Series(close_1d).shift(1).diff().abs()
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # ATR(14) and ATR(50)
    atr_14 = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_50 = pd.Series(tr_1d).ewm(alpha=1/50, adjust=False, min_periods=50).mean().values
    
    # ATR ratio (avoid division by zero)
    atr_ratio = np.where(atr_50 > 0, atr_14 / atr_50, 0)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    volatility_expansion = atr_ratio_aligned > 1.2
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for ATR(50))
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Calculate 4h ATR for stoploss
    tr1_4h = pd.Series(high).diff()
    tr2_4h = pd.Series(low).diff().abs()
    tr3_4h = pd.Series(close).shift(1).diff().abs()
    tr_4h = pd.concat([tr1_4h, tr2_4h, tr3_4h], axis=1).max(axis=1)
    atr_4h_raw = pd.Series(tr_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i]) or
            np.isnan(volume_spike[i]) or np.isnan(volatility_expansion[i]) or np.isnan(atr_4h_raw[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        vol_exp = volatility_expansion[i]
        atr_val = atr_4h_raw[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price crosses below Donchian midpoint
            if price < donchian_mid[i]:
                exit_signal = True
            # ATR-based stoploss: 2*ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price crosses above Donchian midpoint
            if price > donchian_mid[i]:
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
            # LONG: Price breaks above Donchian high AND volume spike AND volatility expansion
            if price > donchian_high[i] and vol_spike and vol_exp:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below Donchian low AND volume spike AND volatility expansion
            elif price < donchian_low[i] and vol_spike and vol_exp:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_Donchian20_1dVolSpike_ATRRat_V1"
timeframe = "4h"
leverage = 1.0