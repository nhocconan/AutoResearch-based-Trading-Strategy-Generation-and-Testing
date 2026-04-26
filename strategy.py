#!/usr/bin/env python3
"""
6h_ElderRay_Regime_Breakout
Hypothesis: Elder Ray (Bull/Bear Power) combined with 1d regime filter (ADX) and 6h Donchian breakout.
In bull regime (1d ADX>25 + price>SMA50): long on 6h Donchian breakout when Bull Power>0.
In bear regime (1d ADX>25 + price<SMA50): short on 6h Donchian breakdown when Bear Power<0.
In range regime (1d ADX<20): fade at Donchian extremes with volume confirmation.
Uses discrete sizing (0.25) to control fees. Target: 50-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need warmup for indicators
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for HTF regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d indicators for regime
    close_1d = pd.Series(df_1d['close'])
    high_1d = pd.Series(df_1d['high'])
    low_1d = pd.Series(df_1d['low'])
    
    # 1d EMA50 for trend
    ema_50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    # 1d ADX for regime strength
    plus_dm = high_1d.diff()
    minus_dm = low_1d.diff().multiply(-1)
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr = pd.concat([
        high_1d - low_1d,
        (high_1d - close_1d.shift()).abs(),
        (low_1d - close_1d.shift()).abs()
    ], axis=1).max(axis=1)
    atr_1d = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    plus_di_1d = 100 * (plus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr_1d)
    minus_di_1d = 100 * (minus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr_1d)
    dx_1d = (abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)).replace([np.inf, -np.inf], 0).fillna(0) * 100
    adx_1d = dx_1d.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1d indicators to 6h
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 6h indicators
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    # Volume confirmation: 20-period volume SMA
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup
    start_idx = max(20, 13, 50)
    
    for i in range(start_idx, n):
        # Regime classification from 1d
        adx_val = adx_1d_aligned[i]
        ema_50_val = ema_50_1d_aligned[i]
        close_val = close[i]
        
        # Determine regime
        is_trending = adx_val > 25
        is_bull_trend = is_trending and (close_val > ema_50_val)
        is_bear_trend = is_trending and (close_val < ema_50_val)
        is_ranging = adx_val < 20
        
        # 6h values
        dh = donchian_high[i]
        dl = donchian_low[i]
        bp = bull_power[i]
        br = bear_power[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        
        # Skip if data not ready
        if np.isnan(dh) or np.isnan(dl) or np.isnan(bp) or np.isnan(br) or np.isnan(avg_vol):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Volume confirmation
        volume_confirmed = vol > 1.5 * avg_vol
        
        if is_bull_trend:
            # Bull regime: long on Donchian breakout with Bull Power > 0
            long_condition = (close[i] > dh) and (bp > 0) and volume_confirmed
            # Exit: Donchian breakdown or Bull Power <= 0
            exit_long = (close[i] < dl) or (bp <= 0)
            
            if long_condition and position != 1:
                signals[i] = base_size
                position = 1
            elif position == 1 and exit_long:
                signals[i] = 0.0
                position = 0
            else:
                # Hold
                signals[i] = base_size if position == 1 else (0.0 if position == 0 else -base_size)
                
        elif is_bear_trend:
            # Bear regime: short on Donchian breakdown with Bear Power < 0
            short_condition = (close[i] < dl) and (br < 0) and volume_confirmed
            # Exit: Donchian breakout or Bear Power >= 0
            exit_short = (close[i] > dh) or (br >= 0)
            
            if short_condition and position != -1:
                signals[i] = -base_size
                position = -1
            elif position == -1 and exit_short:
                signals[i] = 0.0
                position = 0
            else:
                # Hold
                signals[i] = -base_size if position == -1 else (0.0 if position == 0 else base_size)
                
        else:  # ranging regime (ADX < 20) or transition
            # Range: fade at Donchian extremes with volume confirmation
            # Long near lower band, short near upper band
            long_condition = (close[i] <= dl * 1.001) and (bp > 0) and volume_confirmed  # near DL
            short_condition = (close[i] >= dh * 0.999) and (br < 0) and volume_confirmed  # near DH
            
            # Exit: move toward middle or power reversal
            exit_long = (close[i] > (dh + dl) / 2) or (bp <= 0)
            exit_short = (close[i] < (dh + dl) / 2) or (br >= 0)
            
            if long_condition and position != 1:
                signals[i] = base_size
                position = 1
            elif short_condition and position != -1:
                signals[i] = -base_size
                position = -1
            elif position == 1 and exit_long:
                signals[i] = 0.0
                position = 0
            elif position == -1 and exit_short:
                signals[i] = 0.0
                position = 0
            else:
                # Hold
                if position == 1:
                    signals[i] = base_size
                elif position == -1:
                    signals[i] = -base_size
                else:
                    signals[i] = 0.0
    
    return signals

name = "6h_ElderRay_Regime_Breakout"
timeframe = "6h"
leverage = 1.0