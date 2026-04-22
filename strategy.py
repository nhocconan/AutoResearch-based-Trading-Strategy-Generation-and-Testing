#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h EMA(8)/EMA(21) crossover with 4h ADX(14) trend filter and 1d volume confirmation.
# Uses 4h ADX to filter for trending markets (ADX > 25) and 1h EMA crossover for entry timing.
# 1d volume spike confirms institutional interest. Only trades during 08-20 UTC session.
# Designed for 1h timeframe to target 15-35 trades/year per symbol.
# Works in bull/bear via ADX trend filter - only takes trades when trend is strong.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data for ADX trend filter (ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate ADX(14) on 4h data
    # True Range
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    up_move = high_4h - np.roll(high_4h, 1)
    down_move = np.roll(low_4h, 1) - low_4h
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothing (Wilder's smoothing = EMA with alpha=1/period)
    def WilderMA(arr, period):
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) >= period:
            # First value is simple average
            result[period-1] = np.nansum(arr[:period]) / period
            # Subsequent values: Wilder smoothing
            alpha = 1.0 / period
            for i in range(period, len(arr)):
                result[i] = alpha * arr[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_4h = WilderMA(tr, 14)
    plus_di_4h = 100 * WilderMA(plus_dm, 14) / np.where(atr_4h == 0, 1, atr_4h)
    minus_di_4h = 100 * WilderMA(minus_dm, 14) / np.where(atr_4h == 0, 1, atr_4h)
    dx_4h = 100 * np.abs(plus_di_4h - minus_di_4h) / np.where((plus_di_4h + minus_di_4h) == 0, 1, (plus_di_4h + minus_di_4h))
    adx_4h = WilderMA(dx_4h, 14)
    
    # Load 1d data for volume confirmation (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # 1d volume MA(20) for spike detection
    vol_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > 1.5 * vol_ma20_1d  # Volume spike threshold
    
    # Align 4h ADX and 1d volume spike to 1h timeframe
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # 1h EMA(8) and EMA(21) for entry timing
    ema_8 = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready or outside session
        if (np.isnan(adx_4h_aligned[i]) or np.isnan(vol_spike_1d_aligned[i]) or 
            np.isnan(ema_8[i]) or np.isnan(ema_21[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: EMA(8) crosses above EMA(21) + ADX > 25 (strong trend) + volume spike
            if (ema_8[i] > ema_21[i] and ema_8[i-1] <= ema_21[i-1] and
                adx_4h_aligned[i] > 25 and vol_spike_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: EMA(8) crosses below EMA(21) + ADX > 25 (strong trend) + volume spike
            elif (ema_8[i] < ema_21[i] and ema_8[i-1] >= ema_21[i-1] and
                  adx_4h_aligned[i] > 25 and vol_spike_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit on EMA(8) cross below EMA(21) or ADX weakening
                if (ema_8[i] < ema_21[i] and ema_8[i-1] >= ema_21[i-1]) or adx_4h_aligned[i] < 20:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                # Exit on EMA(8) cross above EMA(21) or ADX weakening
                if (ema_8[i] > ema_21[i] and ema_8[i-1] <= ema_21[i-1]) or adx_4h_aligned[i] < 20:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals

name = "1h_EMA8_21_Crossover_4hADX25_1dVolumeSpike"
timeframe = "1h"
leverage = 1.0