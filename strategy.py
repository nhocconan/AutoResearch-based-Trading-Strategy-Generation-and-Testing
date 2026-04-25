#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1wTrendFilter_VolumeConfirm
Hypothesis: Elder Ray (Bull/Bear Power) on 6h with 1-week EMA34 trend filter and volume confirmation captures institutional momentum in both bull and bear markets. Uses discrete sizing (0.25) and ATR-based stoploss (2.0) to manage risk and minimize fee drag. Targets 12-30 trades/year by requiring confluence of Elder Ray signal, weekly trend, and volume spike (>1.8x 20-bar avg).
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
    
    # Get 1-week data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA34 on 1w for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate EMA13 on 6h for Elder Ray (Bull/Bear Power)
    ema_13_6h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13_6h
    bear_power = low - ema_13_6h
    
    # Volume average (20-period) for volume spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) on 6h for stoploss
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(34, 13, 20, 14)  # EMA34_1w, EMA13_6h, vol MA, ATR
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(ema_13_6h[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(vol_ma[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get values
        ema_trend = ema_34_1w_aligned[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        atr_val = atr[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        ema13_val = ema_13_6h[i]
        
        # Volume spike condition: current volume > 1.8x 20-period average
        volume_spike = vol_val > 1.8 * vol_ma_val
        
        if position == 0:
            # Look for entry signals: Elder Ray with weekly trend and volume
            # Long: Bull Power > 0 (bulls in control) + price > weekly EMA34 (uptrend) + volume spike
            long_signal = (bull_val > 0) and (close_val > ema_trend) and volume_spike
            # Short: Bear Power < 0 (bears in control) + price < weekly EMA34 (downtrend) + volume spike
            short_signal = (bear_val < 0) and (close_val < ema_trend) and volume_spike
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions:
            # 1. Stoploss: price moves against position by 2.0*ATR
            if close_val < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Trend reversal: price crosses below weekly EMA34
            elif close_val < ema_trend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 3. Elder Ray exhaustion: Bull Power turns negative
            elif bull_val <= 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Stoploss: price moves against position by 2.0*ATR
            if close_val > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Trend reversal: price crosses above weekly EMA34
            elif close_val > ema_trend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 3. Elder Ray exhaustion: Bear Power turns positive
            elif bear_val >= 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
    
    return signals

name = "6h_ElderRay_BullBearPower_1wTrendFilter_VolumeConfirm"
timeframe = "6h"
leverage = 1.0