#!/usr/bin/env python3
"""
6h_ElderRay_Regime_Breakout
Hypothesis: Elder Ray (Bull Power/Bear Power) with ADX regime filter on 6h, using 1d EMA50 for trend alignment and volume confirmation (>1.5x average volume). 
Targets breakouts in the direction of the 1d trend with Elder Ray momentum confirmation. Designed for low trade frequency (12-30/year) to minimize fee drag.
Works in bull markets via long breakouts with positive Bull Power and in bear markets via short breakouts with negative Bear Power.
Uses discrete position sizing (0.25) to reduce churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need warmup for indicators
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(21) for Elder Ray and stops
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=21, min_periods=21).mean().values
    
    # Calculate average volume for confirmation (20-period SMA)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # ADX(14) for regime filter
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), np.maximum(np.roll(low, 1) - low, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    tr_rma = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / tr_rma
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / tr_rma
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    base_size = 0.25
    atr_multiplier = 3.0  # ATR stoploss multiplier
    
    # Start after warmup (need 21 for ATR, 13 for EMA13, 14 for ADX, 50 for 1d EMA)
    start_idx = max(21, 13, 14, 50)
    
    for i in range(start_idx, n):
        # Hold current position by default
        if position == 0:
            signals[i] = 0.0
        elif position == 1:
            signals[i] = base_size
        else:
            signals[i] = -base_size
        
        # Skip if any data not ready
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(avg_volume[i]) or \
           np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(adx[i]):
            continue
        
        close_val = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_val = ema_50_1d_aligned[i]
        atr_val = atr[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        adx_val = adx[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirmed = vol > 1.5 * avg_vol
        # ADX regime filter: trending market (ADX > 25)
        trending = adx_val > 25
        
        # Long logic: price above 1d EMA50, Bull Power > 0, volume confirmed, trending
        long_condition = (close_val > ema_val) and (bull_val > 0) and volume_confirmed and trending
        # Short logic: price below 1d EMA50, Bear Power < 0, volume confirmed, trending
        short_condition = (close_val < ema_val) and (bear_val < 0) and volume_confirmed and trending
        
        # ATR-based stoploss
        if position == 1:
            stop_price = entry_price - atr_multiplier * atr_val
            if close_val < stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:
            stop_price = entry_price + atr_multiplier * atr_val
            if close_val > stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
            entry_price = close_val  # Enter at next bar open, approximate with close
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
            entry_price = close_val  # Enter at next bar open, approximate with close
        elif position == 1 and (close_val < ema_val or bull_val <= 0):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close_val > ema_val or bear_val >= 0):
            signals[i] = 0.0
            position = 0
    
    return signals

name = "6h_ElderRay_Regime_Breakout"
timeframe = "6h"
leverage = 1.0