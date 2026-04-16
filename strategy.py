#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d ATR filter and volume confirmation.
# Long when price breaks above Donchian(20) high AND 1d ATR ratio (ATR14/ATR50) < 0.8 (low vol regime)
# AND 1d volume > 1.2x 20-period average. Short when price breaks below Donchian(20) low
# AND same conditions. Exit on opposite Donchian break or ATR(14) stoploss (2.5x).
# Uses discrete position size 0.28. Volume + low volatility filter reduces whipsaw in bear markets.
# Target: 60-120 total trades over 4 years (15-30/year).

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: Donchian Channel (20-period) ===
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h ATR for stoploss ===
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr_6h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_6h_raw = pd.Series(tr_6h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # === 1d Indicators: ATR Ratio (ATR14/ATR50) and Volume Spike ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    vol_1d = df_1d['volume'].values
    
    # ATR14 and ATR50 on 1d
    tr1_1d = pd.Series(high_1d).diff()
    tr2_1d = pd.Series(low_1d).diff().abs()
    tr3_1d = pd.Series(close_1d).shift(1).diff().abs()
    tr_1d = pd.concat([tr1_1d, tr2_1d, tr3_1d], axis=1).max(axis=1)
    atr14_1d = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr50_1d = pd.Series(tr_1d).ewm(alpha=1/50, adjust=False, min_periods=50).mean().values
    atr_ratio_1d = atr14_1d / atr50_1d  # < 0.8 = low volatility regime
    
    # Volume confirmation: > 1.2x 20-period average
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = vol_1d > (1.2 * vol_ma_1d)
    
    # Align 1d indicators to 6h
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for ATR50)
    warmup = 60
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or np.isnan(atr_6h_raw[i]) or
            np.isnan(atr_ratio_aligned[i]) or np.isnan(volume_spike_aligned[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike_aligned[i] > 0.5  # convert back to boolean
        low_vol_regime = atr_ratio_aligned[i] < 0.8
        atr_val = atr_6h_raw[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price breaks below Donchian low (opposite breakout)
            if price < low_roll[i]:
                exit_signal = True
            # ATR-based stoploss: 2.5*ATR below entry
            elif price < entry_price - 2.5 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price breaks above Donchian high (opposite breakout)
            if price > high_roll[i]:
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
            # Need both volume spike AND low volatility regime
            if vol_spike and low_vol_regime:
                # LONG: Price breaks above Donchian high
                if price > high_roll[i]:
                    signals[i] = 0.28
                    position = 1
                    entry_price = price
                # SHORT: Price breaks below Donchian low
                elif price < low_roll[i]:
                    signals[i] = -0.28
                    position = -1
                    entry_price = price
        
        else:
            signals[i] = position * 0.28
    
    return signals

name = "6h_Donchian20_1dATR_Volume_LowVol_V1"
timeframe = "6h"
leverage = 1.0