#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla R3/S3 breakout with 1d volume spike confirmation and ATR volatility filter.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for volume spike detection (volume > 2.0 * 20-day average) and ATR-based volatility regime.
- Camarilla levels from 1d: R3/S3 as breakout triggers, R4/S4 as stop/reversal levels.
- Entry: Long when close breaks above R3 with volume spike AND ATR(14) > ATR(50) (high volatility regime).
         Short when close breaks below S3 with volume spike AND ATR(14) > ATR(50).
- Exit: Close below R3 for longs, above S3 for shorts, or volatility collapse (ATR(14) < 0.5 * ATR(50)).
- Signal size: 0.25 discrete to minimize fee drag.
- Works in both bull and bear markets by trading breakouts only during high volatility regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Camarilla levels (R3, S3, R4, S4)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # Need at least 2 days for Camarilla calculation
        return np.zeros(n)
    
    # Camarilla calculation uses previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_r3 = np.zeros_like(close_1d)
    camarilla_s3 = np.zeros_like(close_1d)
    camarilla_r4 = np.zeros_like(close_1d)
    camarilla_s4 = np.zeros_like(close_1d)
    
    for i in range(1, len(df_1d)):
        # Previous day's OHLC
        ph = high_1d[i-1]
        pl = low_1d[i-1]
        pc = close_1d[i-1]
        
        # Camarilla formulas
        camarilla_r3[i] = pc + (ph - pl) * 1.1 / 4
        camarilla_s3[i] = pc - (ph - pl) * 1.1 / 4
        camarilla_r4[i] = pc + (ph - pl) * 1.1 / 2
        camarilla_s4[i] = pc - (ph - pl) * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Calculate 1d volume average for confirmation (20-period)
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate ATR(14) and ATR(50) on 6h for volatility regime filter
    if len(close) < 50:
        return np.zeros(n)
    
    # True Range calculation
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 50 for ATR50, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(atr14[i]) or np.isnan(atr50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volatility regime filter: ATR(14) > ATR(50) indicates high volatility
        high_vol_regime = atr14[i] > atr50[i]
        low_vol_regime = atr14[i] < 0.5 * atr50[i]  # Volatility collapse for exit
        
        # Volume confirmation: current volume > 2.0 * 20-period average volume
        volume_confirm = curr_volume > 2.0 * vol_ma_20_1d_aligned[i] if not np.isnan(vol_ma_20_1d_aligned[i]) else False
        
        # Current Camarilla levels
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        r4 = camarilla_r4_aligned[i]
        s4 = camarilla_s4_aligned[i]
        
        # Exit conditions
        if position != 0:
            # Exit long: close below R3 OR volatility collapse
            if position == 1:
                if curr_close < r3 or low_vol_regime:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: close above S3 OR volatility collapse
            elif position == -1:
                if curr_close > s3 or low_vol_regime:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with volume and volatility filters
        if position == 0:
            # Long: close breaks above R3 with volume spike AND high volatility regime
            long_condition = (curr_close > r3) and volume_confirm and high_vol_regime
            
            # Short: close breaks below S3 with volume spike AND high volatility regime
            short_condition = (curr_close < s3) and volume_confirm and high_vol_regime
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_1dVolSpike_ATRVolatilityFilter_v1"
timeframe = "6h"
leverage = 1.0