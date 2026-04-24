#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d ADX regime filter and volume confirmation.
- Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low (using 13-period EMA)
- Regime filter: ADX > 25 on 1d indicates trending market (use Elder Ray signals)
- ADX <= 25 indicates ranging market (fade Elder Ray extremes)
- Volume confirmation: current volume > 1.5 * 20-period average
- Long in trending: Bull Power > 0 AND ADX > 25
- Short in trending: Bear Power > 0 AND ADX > 25
- Long in ranging: Bull Power < -0.5 * ATR(10) AND ADX <= 25 (oversold)
- Short in ranging: Bear Power < -0.5 * ATR(10) AND ADX <= 25 (overbought)
- Uses 6h primary with 1d HTF to target 50-150 trades over 4 years (12-37/year)
- Elder Ray measures price strength relative to EMA; ADX filters regime; volume confirms momentum
- Signal size: 0.25 discrete levels to minimize fee churn
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
    
    # Get 1d data for HTF indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA13 for Elder Ray
    daily_close = df_1d['close'].values
    ema_13_1d = pd.Series(daily_close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 1d Elder Ray components
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    bull_power = daily_high - ema_13_1d  # High - EMA13
    bear_power = ema_13_1d - daily_low   # EMA13 - Low
    
    # Calculate 1d ATR(10) for ranging thresholds
    daily_tr1 = np.abs(daily_high[1:] - daily_low[:-1])
    daily_tr2 = np.abs(daily_high[1:] - daily_close[:-1])
    daily_tr3 = np.abs(daily_low[1:] - daily_close[:-1])
    daily_tr = np.concatenate([[np.nan], np.maximum(daily_tr1, np.maximum(daily_tr2, daily_tr3))])
    atr_10_1d = pd.Series(daily_tr).rolling(window=10, min_periods=10).mean().values
    
    # Calculate 1d ADX(14) for regime filter
    # +DM and -DM
    daily_up_move = daily_high[1:] - daily_high[:-1]
    daily_down_move = daily_low[:-1] - daily_low[1:]
    plus_dm = np.where((daily_up_move > daily_down_move) & (daily_up_move > 0), daily_up_move, 0)
    minus_dm = np.where((daily_down_move > daily_up_move) & (daily_down_move > 0), daily_down_move, 0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # True Range
    daily_tr1 = np.abs(daily_high[1:] - daily_low[:-1])
    daily_tr2 = np.abs(daily_high[1:] - daily_close[:-1])
    daily_tr3 = np.abs(daily_low[1:] - daily_close[:-1])
    daily_tr = np.concatenate([[np.nan], np.maximum(daily_tr1, np.maximum(daily_tr2, daily_tr3))])
    
    # Smoothed values
    tr_14 = pd.Series(daily_tr).rolling(window=14, min_periods=14).mean().values
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # DI+ and DI-
    plus_di = 100 * plus_dm_14 / tr_14
    minus_di = 100 * minus_dm_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align all 1d indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    atr_10_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_10_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: volume > 1.5 * 20-period average (6h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need ADX, ATR, EMA with sufficient lookback
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(atr_10_1d_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bull_power_val = bull_power_aligned[i]
        bear_power_val = bear_power_aligned[i]
        atr_val = atr_10_1d_aligned[i]
        adx_val = adx_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Determine regime: trending (ADX > 25) or ranging (ADX <= 25)
            if adx_val > 25:
                # Trending market: follow Elder Ray direction
                if bull_power_val > 0 and vol_conf:
                    signals[i] = 0.25
                    position = 1
                elif bear_power_val > 0 and vol_conf:
                    signals[i] = -0.25
                    position = -1
            else:
                # Ranging market: fade Elder Ray extremes
                if bull_power_val < (-0.5 * atr_val) and vol_conf:
                    signals[i] = 0.25
                    position = 1
                elif bear_power_val < (-0.5 * atr_val) and vol_conf:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Elder Ray turns negative OR opposite signal in same regime
            if adx_val > 25:
                # Trending: exit when bull power fades
                if bull_power_val <= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                # Ranging: exit when mean reversion complete
                if bull_power_val >= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        elif position == -1:
            # Short exit: Elder Ray turns negative OR opposite signal in same regime
            if adx_val > 25:
                # Trending: exit when bear power fades
                if bear_power_val <= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                # Ranging: exit when mean reversion complete
                if bear_power_val >= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_1dADX_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0