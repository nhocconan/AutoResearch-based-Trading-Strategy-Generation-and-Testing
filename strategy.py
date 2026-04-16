#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR filter and volume confirmation.
# Long when price breaks above Donchian(20) high AND 1d volume > 1.3x 20-period average AND ATR(14) > ATR(50) (volatility expansion).
# Short when price breaks below Donchian(20) low AND 1d volume > 1.3x 20-period average AND ATR(14) > ATR(50).
# Exit on opposite Donchian break or ATR-based stoploss (2.5*ATR from entry).
# Uses discrete position size 0.25. Designed to capture volatility expansion breakouts with institutional volume.
# Works in both bull and bear markets by requiring volume confirmation and volatility filter.
# Target: 75-200 total trades over 4 years (19-50/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: Donchian Channel (20) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1d Indicators: Volume Spike and ATR Regime Filter ===
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.3 * vol_ma_1d_aligned)
    
    # ATR(14) and ATR(50) for volatility regime
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14_raw = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_50_raw = pd.Series(tr_1d).ewm(alpha=1/50, adjust=False, min_periods=50).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14_raw)
    atr_50_aligned = align_htf_to_ltf(prices, df_1d, atr_50_raw)
    vol_expansion = atr_14_aligned > atr_50_aligned  # ATR(14) > ATR(50) indicates volatility expansion
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed)
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(volume_spike[i]) or
            np.isnan(atr_14_aligned[i]) or np.isnan(atr_50_aligned[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        vol_exp = vol_expansion[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price breaks below Donchian low (opposite breakout)
            if price < donchian_low[i]:
                exit_signal = True
            # ATR-based stoploss: 2.5*ATR below entry (using 1d ATR)
            elif price < entry_price - 2.5 * atr_14_aligned[i]:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price breaks above Donchian high (opposite breakout)
            if price > donchian_high[i]:
                exit_signal = True
            # ATR-based stoploss: 2.5*ATR above entry (using 1d ATR)
            elif price > entry_price + 2.5 * atr_14_aligned[i]:
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

name = "4h_Donchian20_1dVolumeSpike_VolExp_V1"
timeframe = "4h"
leverage = 1.0