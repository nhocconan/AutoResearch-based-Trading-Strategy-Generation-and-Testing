#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R (14) with 1d ATR-based volatility filter and volume confirmation.
# Long when %R < -80 (oversold) AND volume > 1.5x 20-period 4h volume average AND 1d ATR ratio (current/10-period MA) > 1.2 (elevated volatility).
# Short when %R > -20 (overbought) AND volume > 1.5x 20-period 4h volume average AND 1d ATR ratio > 1.2.
# Exit when %R crosses -50 (mean reversion completion) or ATR-based stoploss (2*ATR from entry).
# Uses discrete position size 0.25. Designed to capture mean reversion in volatile conditions with volume confirmation.
# Works in both bull and bear markets by requiring volatility expansion and volume, avoiding low-volume ranging markets.
# Target: 75-200 total trades over 4 years (19-50/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: Williams %R (14-period) ===
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # === 4h Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_4h)
    
    # === 1d Indicators: ATR Ratio (current ATR / 10-period MA) for volatility filter ===
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
    atr_ma_1d = pd.Series(atr_1d).rolling(window=10, min_periods=10).mean().values
    atr_ratio = atr_1d / atr_ma_1d
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    volatility_filter = atr_ratio_aligned > 1.2
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for %R/ATR/EMA)
    warmup = 50
    
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
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(volume_spike[i]) or np.isnan(volatility_filter[i]) or
            np.isnan(atr_4h_raw[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        vol_filter = volatility_filter[i]
        wr = williams_r[i]
        atr_val = atr_4h_raw[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Williams %R crosses above -50 (mean reversion completion)
            if wr > -50:
                exit_signal = True
            # ATR-based stoploss: 2*ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Williams %R crosses below -50 (mean reversion completion)
            if wr < -50:
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
            # LONG: Williams %R < -80 (oversold) AND volume spike AND volatility filter
            if wr < -80 and vol_spike and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Williams %R > -20 (overbought) AND volume spike AND volatility filter
            elif wr > -20 and vol_spike and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_WilliamsR_14_4hVolumeSpike_1dATRRatio_V1"
timeframe = "4h"
leverage = 1.0