#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_ChopFilter_v2
Hypothesis: 12h Camarilla pivot R1/S1 breakout with 1-day trend filter and choppiness regime filter.
Only trade in trending markets (ADX > 25) in direction of 1-day EMA34 trend: long R1 breakout in uptrend, short S1 breakdown in downtrend.
Uses choppiness index to avoid ranging markets (CHOP > 61.8 = range, avoid trading).
Designed for 12-37 trades/year (50-150 over 4 years) by requiring confluence of breakout, trend, volume, and regime filter.
Works in bull/bear via 1-day trend filter: only takes long breakouts in uptrend, short in downtrend.
Uses discrete position sizing (0.25) to minimize fee churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF trend and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for HTF trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    htf_trend = np.where(close > ema_34_1d_aligned, 1, -1)  # 1 = uptrend, -1 = downtrend
    
    # Calculate Camarilla pivot levels from 1d data
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    R1_1d = typical_price_1d + (1.1/12) * (df_1d['high'] - df_1d['low'])  # R1 level
    S1_1d = typical_price_1d - (1.1/12) * (df_1d['high'] - df_1d['low'])  # S1 level
    
    # Align Camarilla levels to 12h timeframe
    R1_1d_aligned = align_htf_to_ltf(prices, df_1d, R1_1d.values)
    S1_1d_aligned = align_htf_to_ltf(prices, df_1d, S1_1d.values)
    
    # Calculate ADX for regime filter (using 12h data)
    # TR calculation
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # +DM and -DM
    up_move = high - np.concatenate([[high[0]], high[:-1]])
    down_move = np.concatenate([[low[0]], low[:-1]]) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed TR, +DM, -DM (using Wilder's smoothing)
    def Wilder_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nansum(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr_period = 14
    tr_smooth = Wilder_smoothing(tr, atr_period)
    plus_dm_smooth = Wilder_smoothing(plus_dm, atr_period)
    minus_dm_smooth = Wilder_smoothing(minus_dm, atr_period)
    
    # +DI and -DI
    plus_di = np.where(tr_smooth != 0, (plus_dm_smooth / tr_smooth) * 100, 0)
    minus_di = np.where(tr_smooth != 0, (minus_dm_smooth / tr_smooth) * 100, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, np.abs((plus_di - minus_di) / (plus_di + minus_di)) * 100, 0)
    adx = Wilder_smoothing(dx, atr_period)
    
    # Calculate Choppiness Index (using 12h data)
    def choppiness_index(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
        tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Sum of TR over period
        tr_sum = np.zeros_like(close)
        for i in range(len(close)):
            if i < period:
                tr_sum[i] = np.nan
            else:
                tr_sum[i] = np.nansum(tr[i-period+1:i+1])
        
        # Highest high and lowest low over period
        hh = np.zeros_like(close)
        ll = np.zeros_like(close)
        for i in range(len(close)):
            if i < period:
                hh[i] = np.nan
                ll[i] = np.nan
            else:
                hh[i] = np.nanmax(high[i-period+1:i+1])
                ll[i] = np.nanmin(low[i-period+1:i+1])
        
        # Choppiness Index formula
        chop = np.zeros_like(close)
        for i in range(len(close)):
            if np.isnan(tr_sum[i]) or np.isnan(hh[i]) or np.isnan(ll[i]) or hh[i] == ll[i]:
                chop[i] = np.nan
            else:
                chop[i] = 100 * np.log10(tr_sum[i] / (hh[i] - ll[i])) / np.log10(period)
        return chop
    
    chop = choppiness_index(high, low, close, period=14)
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for 1d EMA, 20 for volume MA, 28 for ADX/CHOP)
    start_idx = max(34, 20, 28)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(R1_1d_aligned[i]) or np.isnan(S1_1d_aligned[i]) or
            np.isnan(adx[i]) or np.isnan(chop[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Regime filter: only trade when ADX > 25 (trending) and CHOP <= 61.8 (not ranging)
        trending_regime = adx[i] > 25
        not_ranging = chop[i] <= 61.8
        regime_ok = trending_regime and not_ranging
        
        # Volume spike condition
        volume_spike = volume[i] > 1.3 * vol_ma_20[i]
        
        # Breakout conditions with trend filter
        if htf_trend[i] == 1:  # Uptrend on 1d
            # Long breakout above R1 with volume spike and regime filter
            if close[i] > R1_1d_aligned[i] and volume_spike and regime_ok:
                if position != 1:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25
            # Exit long if price falls below S1 (reversal signal) or regime breaks
            elif position == 1 and (close[i] < S1_1d_aligned[i] or not regime_ok):
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
        elif htf_trend[i] == -1:  # Downtrend on 1d
            # Short breakdown below S1 with volume spike and regime filter
            if close[i] < S1_1d_aligned[i] and volume_spike and regime_ok:
                if position != -1:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25
            # Exit short if price rises above R1 (reversal signal) or regime breaks
            elif position == -1 and (close[i] > R1_1d_aligned[i] or not regime_ok):
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

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_ChopFilter_v2"
timeframe = "12h"
leverage = 1.0