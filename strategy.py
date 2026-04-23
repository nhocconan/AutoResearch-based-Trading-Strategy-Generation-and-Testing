#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R mean reversion with 1d ADX regime filter and volume confirmation.
Williams %R identifies overbought/oversold conditions (-20 to -80 range). 
ADX > 25 indicates trending market (avoid mean reversion in strong trends), ADX < 20 indicates ranging (favor mean reversion).
Volume spike confirms mean reversion exhaustion. Designed for 4h timeframe to balance trade frequency and signal quality.
Target: 20-50 trades/year per symbol (80-200 total over 4 years).
Uses discrete position sizing (0.25) to minimize fee drag while maintaining profitability.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Replace division by zero with -50 (neutral)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 1d ADX (14-period) for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    tr[0] = high_1d[0] - low_1d[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(values, period):
        result = np.zeros_like(values)
        result[period-1] = np.nansum(values[:period])
        for i in range(period, len(values)):
            result[i] = result[i-1] - (result[i-1] / period) + values[i]
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_1d != 0, dm_plus_smooth / atr_1d * 100, 0)
    di_minus = np.where(atr_1d != 0, dm_minus_smooth / atr_1d * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = wilders_smoothing(dx, 14)
    
    # Align HTF indicators to LTF
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)  # Williams %R needs no extra delay
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)  # ADX needs no extra delay
    
    # Calculate volume spike: current volume > 1.8x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.8 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # need Williams %R and volume MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: ADX < 25 indicates ranging market (favor mean reversion)
        ranging_market = adx_aligned[i] < 25
        
        # Williams %R signals: oversold < -80, overbought > -20
        oversold = williams_r_aligned[i] < -80
        overbought = williams_r_aligned[i] > -20
        
        if position == 0:
            # Long: Oversold AND ranging market AND volume spike (exhaustion selling)
            if oversold and ranging_market and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Overbought AND ranging market AND volume spike (exhaustion buying)
            elif overbought and ranging_market and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Williams %R returns to neutral range (-50 center) or opposite extreme
            exit_signal = False
            if position == 1:
                # Exit long when Williams %R returns above -50 (neutral) or reaches overbought
                if williams_r_aligned[i] > -50:
                    exit_signal = True
            elif position == -1:
                # Exit short when Williams %R returns below -50 (neutral) or reaches oversold
                if williams_r_aligned[i] < -50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_WilliamsR_MeanReversion_1dADXRegime_VolumeConfirmation"
timeframe = "4h"
leverage = 1.0