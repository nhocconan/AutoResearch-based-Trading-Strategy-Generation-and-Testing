#!/usr/bin/env python3
"""
6h_Camarilla_R4_S4_Breakout_1wTrend_1dRegime_v1
Hypothesis: 6h Camarilla pivot R4/S4 breakout with 1-week trend filter and 1-day regime filter.
Only trade breakouts in direction of weekly trend (long R4 breakout in weekly uptrend, short S4 breakdown in weekly downtrend).
Use 1-day ADX and volatility regime to avoid choppy markets and whipsaws.
Designed for 12-37 trades/year (50-150 over 4 years) by requiring confluence of weekly trend, breakout, and daily regime filter.
Works in bull/bear via weekly trend filter: only takes long breakouts in weekly uptrend, short in weekly downtrend.
Uses discrete position sizing (0.25) to minimize fee churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w and 1d data ONCE before loop for HTF filters
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1w EMA34 for weekly trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    weekly_trend = np.where(close > ema_34_1w_aligned, 1, -1)  # 1 = uptrend, -1 = downtrend
    
    # Calculate Camarilla pivot levels from 1d data
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    R4_1d = typical_price_1d + (1.1/2) * (df_1d['high'] - df_1d['low'])  # R4 level
    S4_1d = typical_price_1d - (1.1/2) * (df_1d['high'] - df_1d['low'])  # S4 level
    
    # Align Camarilla levels to 6h timeframe
    R4_1d_aligned = align_htf_to_ltf(prices, df_1d, R4_1d.values)
    S4_1d_aligned = align_htf_to_ltf(prices, df_1d, S4_1d.values)
    
    # Calculate 1d ADX for regime filter
    # TR calculation
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr3 = np.abs(df_1d['low'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # +DM and -DM
    up_move_1d = df_1d['high'] - np.concatenate([[df_1d['high'].iloc[0]], df_1d['high'].iloc[:-1]])
    down_move_1d = np.concatenate([[df_1d['low'].iloc[0]], df_1d['low'].iloc[:-1]]) - df_1d['low']
    plus_dm_1d = np.where((up_move_1d > down_move_1d) & (up_move_1d > 0), up_move_1d, 0)
    minus_dm_1d = np.where((down_move_1d > up_move_1d) & (down_move_1d > 0), down_move_1d, 0)
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing)
    def Wilder_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nansum(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr_period = 14
    tr_smooth_1d = Wilder_smoothing(tr_1d.values, atr_period)
    plus_dm_smooth_1d = Wilder_smoothing(plus_dm_1d, atr_period)
    minus_dm_smooth_1d = Wilder_smoothing(minus_dm_1d, atr_period)
    
    # +DI and -DI
    plus_di_1d = np.where(tr_smooth_1d != 0, (plus_dm_smooth_1d / tr_smooth_1d) * 100, 0)
    minus_di_1d = np.where(tr_smooth_1d != 0, (minus_dm_smooth_1d / tr_smooth_1d) * 100, 0)
    
    # DX and ADX
    dx_1d = np.where((plus_di_1d + minus_di_1d) != 0, np.abs((plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)) * 100, 0)
    adx_1d = Wilder_smoothing(dx_1d, atr_period)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 1d volatility regime (using ATR ratio)
    atr_14_1d = Wilder_smoothing(tr_1d.values, atr_period)
    atr_50_1d = Wilder_smoothing(tr_1d.values, 50)
    atr_ratio_1d = atr_14_1d / (atr_50_1d + 1e-10)
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average (6h data)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for 1w EMA, 20 for volume MA, 50 for ATR ratio)
    start_idx = max(34, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(R4_1d_aligned[i]) or np.isnan(S4_1d_aligned[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(atr_ratio_1d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Regime filter: only trade when ADX > 20 (trending) and ATR ratio < 1.8 (not extreme volatility)
        trending_regime = adx_1d_aligned[i] > 20
        normal_volatility = atr_ratio_1d_aligned[i] < 1.8
        regime_ok = trending_regime and normal_volatility
        
        # Volume spike condition
        volume_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        # Breakout conditions with weekly trend filter
        if weekly_trend[i] == 1:  # Weekly uptrend
            # Long breakout above R4 with volume spike and regime filter
            if close[i] > R4_1d_aligned[i] and volume_spike and regime_ok:
                if position != 1:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25
            # Exit long if price falls below S4 (reversal signal) or regime breaks
            elif position == 1 and (close[i] < S4_1d_aligned[i] or not regime_ok):
                signals[i] = 0.0
                position = 0
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
        elif weekly_trend[i] == -1:  # Weekly downtrend
            # Short breakdown below S4 with volume spike and regime filter
            if close[i] < S4_1d_aligned[i] and volume_spike and regime_ok:
                if position != -1:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25
            # Exit short if price rises above R4 (reversal signal) or regime breaks
            elif position == -1 and (close[i] > R4_1d_aligned[i] or not regime_ok):
                signals[i] = 0.0
                position = 0
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
        else:
            # Should not happen with our trend calculation
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R4_S4_Breakout_1wTrend_1dRegime_v1"
timeframe = "6h"
leverage = 1.0