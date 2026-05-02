#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w ADX regime filter and volume confirmation
# Donchian(20) breakout provides clear price channel structure for trend following
# 1w ADX > 25 filters for trending regimes only, avoiding whipsaws in ranging markets
# Volume confirmation (>1.5 x 20-period EMA) validates breakout strength
# Works in bull markets (long on upper breakout) and bear markets (short on lower breakout)
# Uses discrete position sizing (0.25) to minimize fee churn and control drawdown
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag
# 12h timeframe balances trade frequency and signal quality per research

name = "12h_Donchian20_Breakout_1wADX_Regime_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h Donchian(20) channels
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1w data for ADX regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # 1w ADX calculation (14-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[:-1])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = np.concatenate([[np.nan], high_1w[1:] - high_1w[:-1]])
    down_move = np.concatenate([[np.nan], low_1w[:-1] - low_1w[1:]])
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values (Wilder's smoothing)
    def WilderSmooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[1:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_1w = WilderSmooth(tr, 14)
    plus_di_1w = 100 * WilderSmooth(plus_dm, 14) / atr_1w
    minus_di_1w = 100 * WilderSmooth(minus_dm, 14) / atr_1w
    dx_1w = 100 * np.abs(plus_di_1w - minus_di_1w) / (plus_di_1w + minus_di_1w)
    adx_1w = WilderSmooth(dx_1w, 14)
    
    # Align 1w ADX to 12h timeframe (wait for completed 1w bar)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Volume confirmation (volume spike > 1.5 x 20-period EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for Donchian and ADX)
    start_idx = max(20, 34)  # 20 for Donchian, 34 for ADX (14+14+6 for Wilder smoothing)
    
    for i in range(start_idx, n):
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(adx_1w_aligned[i]) or np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Determine regime from 1w ADX
        trending = adx_1w_aligned[i] > 25
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above upper Donchian with volume confirmation and trending regime
            if close[i] > highest_20[i] and volume_confirmation[i] and trending:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian with volume confirmation and trending regime
            elif close[i] < lowest_20[i] and volume_confirmation[i] and trending:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below lower Donchian (contrarian exit) OR regime changes to ranging
            if close[i] < lowest_20[i] or not trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above upper Donchian (contrarian exit) OR regime changes to ranging
            if close[i] > highest_20[i] or not trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals