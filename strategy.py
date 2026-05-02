#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 12h ADX regime filter and volume confirmation
# Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
# 12h ADX > 25 indicates trending market, ADX < 20 indicates ranging
# Long when Bull Power > 0 AND ADX > 25 (strong uptrend) AND volume spike
# Short when Bear Power < 0 AND ADX > 25 (strong downtrend) AND volume spike
# Exit when power reverses or ADX < 20 (range) OR opposing power signal
# Uses discrete position sizing (0.25) to minimize fee churn
# Targets 12-37 trades/year (50-150 total over 4 years) for 6h timeframe

name = "6h_ElderRay_12hADX_Regime_VolumeConfirm_v1"
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
    
    # Load 12h data ONCE before loop for ADX regime filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h ADX for regime filter
    # ADX calculation: +DI, -DI, DX, then ADX
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
    atr_period = 14
    alpha = 1.0 / atr_period
    atr = np.full_like(tr, np.nan)
    dm_plus_smooth = np.full_like(dm_plus, np.nan)
    dm_minus_smooth = np.full_like(dm_minus, np.nan)
    
    # Initialize first values
    if not np.isnan(tr[atr_period]):
        atr[atr_period] = np.nanmean(tr[1:atr_period+1])
        dm_plus_smooth[atr_period] = np.nanmean(dm_plus[1:atr_period+1])
        dm_minus_smooth[atr_period] = np.nanmean(dm_minus[1:atr_period+1])
    
    # Wilder's smoothing
    for i in range(atr_period + 1, len(tr)):
        if np.isnan(atr[i-1]):
            atr[i] = np.nan
        else:
            atr[i] = (atr[i-1] * (atr_period - 1) + tr[i]) / atr_period
        if np.isnan(dm_plus_smooth[i-1]):
            dm_plus_smooth[i] = np.nan
        else:
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (atr_period - 1) + dm_plus[i]) / atr_period
        if np.isnan(dm_minus_smooth[i-1]):
            dm_minus_smooth[i] = np.nan
        else:
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (atr_period - 1) + dm_minus[i]) / atr_period
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = np.full_like(dx, np.nan)
    adx_period = 14
    adx_alpha = 1.0 / adx_period
    
    # Initialize ADX
    if not np.isnan(dx[adx_period]):
        adx[adx_period] = np.nanmean(dx[1:adx_period+1])
    
    # Wilder's smoothing for ADX
    for i in range(adx_period + 1, len(dx)):
        if np.isnan(adx[i-1]):
            adx[i] = np.nan
        else:
            adx[i] = (adx[i-1] * (adx_period - 1) + dx[i]) / adx_period
    
    # Align 12h ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Calculate 6h EMA(13) for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Calculate volume spike (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for ADX calculation and EMA)
    start_idx = 50  # buffer for ADX and EMA calculations
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(adx_aligned[i]) or np.isnan(ema_13[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Bull Power > 0 AND ADX > 25 (strong uptrend) AND volume spike
            if (bull_power[i] > 0 and 
                adx_aligned[i] > 25 and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND ADX > 25 (strong downtrend) AND volume spike
            elif (bear_power[i] < 0 and 
                  adx_aligned[i] > 25 and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Bull Power <= 0 (trend weakening) OR ADX < 20 (range) OR Bear Power < 0 (reversal)
            if bull_power[i] <= 0 or adx_aligned[i] < 20 or bear_power[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bear Power >= 0 (trend weakening) OR ADX < 20 (range) OR Bull Power > 0 (reversal)
            if bear_power[i] >= 0 or adx_aligned[i] < 20 or bull_power[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals