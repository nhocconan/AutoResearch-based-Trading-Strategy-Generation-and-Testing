#!/usr/bin/env python3
# 4H_CAMARILLA_R1_S1_BREAKOUT_1D_TREND_VOLUME_FILTER
# Hypothesis: Breakouts above daily Camarilla R1 in uptrend (price > EMA34) and breakdowns below S1 in downtrend (price < EMA34), with volume confirmation (volume > 1.8x 30-period average) and ADX > 25 for trend strength. Designed to work in both bull and bear markets by requiring trend alignment and avoiding whipsaw through volume and trend filters. Focus on BTC/ETH with robust risk controls.

name = "4H_CAMARILLA_R1_S1_BREAKOUT_1D_TREND_VOLUME_FILTER"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 1d data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 1d bar (based on previous day's range)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    valid_idx = ~np.isnan(prev_high) & ~np.isnan(prev_low) & ~np.isnan(prev_close)
    camarilla_r1 = np.full_like(prev_close, np.nan)
    camarilla_s1 = np.full_like(prev_close, np.nan)
    
    camarilla_r1[valid_idx] = prev_close[valid_idx] + 1.1 * (prev_high[valid_idx] - prev_low[valid_idx]) / 12
    camarilla_s1[valid_idx] = prev_close[valid_idx] - 1.1 * (prev_high[valid_idx] - prev_low[valid_idx]) / 12
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Get 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 1.8x 30-period average (balanced threshold)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_confirmed = volume > (1.8 * vol_ma)
    
    # ADX for trend strength (14-period)
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), np.maximum(np.roll(low, 1) - low, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_dm14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    plus_di14 = np.where(tr14 != 0, (plus_dm14 / tr14) * 100, 0)
    minus_di14 = np.where(tr14 != 0, (minus_dm14 / tr14) * 100, 0)
    dx = np.where((plus_di14 + minus_di14) != 0, np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14) * 100, 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_strong = adx > 25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    cooldown = 0  # cooldown counter to prevent immediate re-entry
    
    for i in range(50, n):
        # Decrease cooldown if active
        if cooldown > 0:
            cooldown -= 1
        
        if position == 0 and cooldown == 0:
            # LONG: Price breaks above R1 with volume confirmation in uptrend (price > EMA34) and strong trend (ADX > 25)
            if camarilla_r1_aligned[i] > 0 and not np.isnan(camarilla_r1_aligned[i]) and \
               high[i] > camarilla_r1_aligned[i] and volume_confirmed[i] and close[i] > ema_34_1d_aligned[i] and adx_strong[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 with volume confirmation in downtrend (price < EMA34) and strong trend (ADX > 25)
            elif camarilla_s1_aligned[i] > 0 and not np.isnan(camarilla_s1_aligned[i]) and \
                 low[i] < camarilla_s1_aligned[i] and volume_confirmed[i] and close[i] < ema_34_1d_aligned[i] and adx_strong[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses back below R1 or trend weakens (price < EMA34)
            if camarilla_r1_aligned[i] > 0 and not np.isnan(camarilla_r1_aligned[i]) and \
               low[i] < camarilla_r1_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                cooldown = 3  # 3-bar cooldown after exit
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses back above S1 or trend weakens (price > EMA34)
            if camarilla_s1_aligned[i] > 0 and not np.isnan(camarilla_s1_aligned[i]) and \
               high[i] > camarilla_s1_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                cooldown = 3  # 3-bar cooldown after exit
            else:
                signals[i] = -0.25
    
    return signals