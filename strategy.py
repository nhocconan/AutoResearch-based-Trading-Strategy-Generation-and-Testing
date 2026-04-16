#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with 1d volume confirmation and chop regime filter.
# Long when Williams %R(14) crosses above -80 (oversold) AND 1d volume > 1.3x 20-period average AND chop > 61.8 (ranging market).
# Short when Williams %R(14) crosses below -20 (overbought) AND 1d volume > 1.3x 20-period average AND chop > 61.8.
# Exit on opposite Williams %R cross (-50 for long, -50 for short) or ATR stoploss (2*ATR).
# Uses discrete position size 0.25. Designed to capture mean reversion in ranging markets with volume confirmation.
# Works in both bull and bear markets by requiring volume confirmation and chop regime filter.
# Target: 75-200 total trades over 4 years (19-50/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: Williams %R(14) ===
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # === 1d Indicators: Volume Spike (volume > 1.3x 20-period average) ===
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.3 * vol_ma_1d_aligned)
    
    # === 1d Indicators: Choppiness Index (CHOP) regime filter ===
    # CHOP > 61.8 = ranging market (good for mean reversion)
    tr_1d = np.maximum(np.maximum(df_1d['high'].values - df_1d['low'].values,
                                  np.abs(df_1d['high'].values - df_1d['close'].shift(1).values)),
                       np.abs(df_1d['low'].values - df_1d['close'].shift(1).values))
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    chop_denom = np.log14(14) * atr_1d  # Using log base e approximation: ln(14) ≈ 2.639
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)  # Avoid division by zero
    sum_tr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_tr_1d / chop_denom) / np.log10(14)
    chop = np.where(np.isnan(chop), 50, chop)  # Fill NaN with neutral value
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    chop_filter = chop_aligned > 61.8
    
    # === 4h ATR for stoploss ===
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr_4h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_4h_raw = pd.Series(tr_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
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
        if (np.isnan(williams_r[i]) or np.isnan(volume_spike[i]) or np.isnan(chop_filter[i]) or
            np.isnan(atr_4h_raw[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        chop_regime = chop_filter[i]
        atr_val = atr_4h_raw[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Williams %R crosses below -50 (mean reversion complete)
            if williams_r[i] < -50 and williams_r[i-1] >= -50:
                exit_signal = True
            # ATR-based stoploss: 2*ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Williams %R crosses above -50 (mean reversion complete)
            if williams_r[i] > -50 and williams_r[i-1] <= -50:
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
            # Williams %R crossover signals
            williams_cross_up = williams_r[i] > -80 and williams_r[i-1] <= -80
            williams_cross_down = williams_r[i] < -20 and williams_r[i-1] >= -20
            
            # LONG: Williams %R crosses above -80 (oversold) AND volume spike AND chop regime
            if williams_cross_up and vol_spike and chop_regime:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Williams %R crosses below -20 (overbought) AND volume spike AND chop regime
            elif williams_cross_down and vol_spike and chop_regime:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_WilliamsR_1dVolumeSpike_ChopFilter_V1"
timeframe = "4h"
leverage = 1.0