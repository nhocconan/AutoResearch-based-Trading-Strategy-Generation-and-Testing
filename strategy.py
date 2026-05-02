#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d ADX Regime + Volume Confirmation
# Elder Ray measures bull/bear power relative to EMA13. ADX(14) > 25 indicates trending market.
# In strong trends (ADX>25): go long when Bull Power > 0 and rising, short when Bear Power < 0 and falling.
# In ranging markets (ADX<20): fade extremes - short when Bull Power > 0.7*ATR, long when Bear Power < -0.7*ATR.
# Volume spike (2.0x 20-period MA) confirms conviction. Target: 50-150 trades over 4 years on 6h.
# Works in bull/bear/ranging by adapting logic to regime.

name = "6h_ElderRay_ADX_Regime_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d ADX for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    tr_14 = wilders_smoothing(tr, 14)
    dm_plus_14 = wilders_smoothing(dm_plus, 14)
    dm_minus_14 = wilders_smoothing(dm_minus, 14)
    
    # DI and ADX
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    adx = wilders_smoothing(dx, 14)
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 6h EMA13 for Elder Ray
    if len(close) < 13:
        return np.zeros(n)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Calculate 6h ATR(14) for dynamic thresholds in ranging market
    if len(high) < 14 or len(low) < 14 or len(close) < 14:
        return np.zeros(n)
    
    tr_6h1 = high - low
    tr_6h2 = np.abs(high - np.roll(close, 1))
    tr_6h3 = np.abs(low - np.roll(close, 1))
    tr_6h = np.maximum(tr_6h1, np.maximum(tr_6h2, tr_6h3))
    tr_6h[0] = tr_6h1[0]
    atr_14 = pd.Series(tr_6h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = max(30, 13, 14, 20)  # 1d ADX warmup, 6h EMA13, ATR, volume MA
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(adx_aligned[i]) or np.isnan(ema_13[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(atr_14[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        adx_val = adx_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            if adx_val > 25:  # Trending market
                # Long: Bull Power > 0 and rising (strong buying pressure)
                if (bull_power[i] > 0 and 
                    bull_power[i] > bull_power[i-1] and 
                    vol_spike):
                    signals[i] = 0.25
                    position = 1
                # Short: Bear Power < 0 and falling (strong selling pressure)
                elif (bear_power[i] < 0 and 
                      bear_power[i] < bear_power[i-1] and 
                      vol_spike):
                    signals[i] = -0.25
                    position = -1
            else:  # Ranging market (ADX < 25)
                # Fade extremes: short when Bull Power too high, long when Bear Power too low
                if (bull_power[i] > 0.7 * atr_14[i] and 
                    vol_spike):
                    signals[i] = -0.25
                    position = -1
                elif (bear_power[i] < -0.7 * atr_14[i] and 
                      vol_spike):
                    signals[i] = 0.25
                    position = 1
        
        elif position == 1:  # Long position
            # Exit conditions
            exit_signal = False
            if adx_val > 25:  # In trend: exit when power fades
                if bull_power[i] <= 0 or bull_power[i] < bull_power[i-1]:
                    exit_signal = True
            else:  # In range: exit when power normalizes
                if bull_power[i] < 0.3 * atr_14[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions
            exit_signal = False
            if adx_val > 25:  # In trend: exit when power fades
                if bear_power[i] >= 0 or bear_power[i] > bear_power[i-1]:
                    exit_signal = True
            else:  # In range: exit when power normalizes
                if bear_power[i] > -0.3 * atr_14[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals