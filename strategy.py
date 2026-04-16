#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d volume spike and choppiness regime filter.
# Long when price closes above Camarilla R3 AND 1d volume > 1.5x 20-period average AND chop > 61.8 (range).
# Short when price closes below Camarilla S3 AND 1d volume > 1.5x 20-period average AND chop > 61.8 (range).
# Exit on opposite Camarilla level (R3/S3) or ATR-based stoploss (2*ATR from entry).
# Uses discrete position size 0.25. Designed to capture mean-reversion breaks in ranging markets with volume confirmation.
# Works in both bull and bear markets by requiring chop regime filter (range-bound) and volume confirmation, avoiding false breakouts.
# Target: 75-200 total trades over 4 years (19-50/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: Camarilla Pivot Levels (R3, S3) ===
    # Camarilla: R3 = close + (high - low) * 1.1/4, S3 = close - (high - low) * 1.1/4
    camarilla_r3 = close + (high - low) * 1.1 / 4
    camarilla_s3 = close - (high - low) * 1.1 / 4
    
    # === 1d Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.5 * vol_ma_1d_aligned)
    
    # === 1d Indicators: Choppiness Index (CHOP > 61.8 = ranging) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    atr_1d = pd.Series(high_1d - low_1d).rolling(window=14, min_periods=14).mean().values
    sum_tr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    max_hh_ll_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values - pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_1d = 100 * np.log10(sum_tr_14 / (np.log(14) * max_hh_ll_14)) / np.log10(14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    chop_range = chop_1d_aligned > 61.8  # ranging market
    
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
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for chop/volume/ATR)
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or np.isnan(volume_spike[i]) or
            np.isnan(chop_range[i]) or np.isnan(atr_4h_raw[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        chop_reg = chop_range[i]
        atr_val = atr_4h_raw[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price closes below Camarilla S3
            if price < camarilla_s3[i]:
                exit_signal = True
            # ATR-based stoploss: 2*ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price closes above Camarilla R3
            if price > camarilla_r3[i]:
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
            # LONG: Price closes above Camarilla R3 AND volume spike AND chop > 61.8 (ranging)
            if price > camarilla_r3[i] and vol_spike and chop_reg:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price closes below Camarilla S3 AND volume spike AND chop > 61.8 (ranging)
            elif price < camarilla_s3[i] and vol_spike and chop_reg:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_CamarillaR3S3_1dVolumeSpike_ChopFilter_V1"
timeframe = "4h"
leverage = 1.0