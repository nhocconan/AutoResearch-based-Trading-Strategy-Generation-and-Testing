#!/usr/bin/env python3
# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) combined with ADX regime filter and volume confirmation.
# Elder Ray measures bullish/bearish power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13.
# In strong trends (ADX > 25): go long when Bull Power > 0 and rising, short when Bear Power < 0 and falling.
# In ranging markets (ADX < 20): fade extremes - long when Bear Power < -0.5*ATR and turning up, short when Bull Power > 0.5*ATR and turning down.
# Volume confirmation requires 1.5x average volume to avoid false signals.
# Uses ATR(14) trailing stop (2.0x) for risk control. Discrete sizing 0.25.
# Target: 50-150 total trades over 4 years (12-37/year) on 6h.
# Works in both bull (trend following) and bear (mean reversion in ranges) markets via regime adaptation.

name = "6h_ElderRay_ADX_Regime_VolumeSpike_ATRStop_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for stops and regime filters
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate EMA13 for Elder Ray
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema13  # Bull Power: High - EMA13
    bear_power = low - ema13   # Bear Power: Low - EMA13
    
    # Calculate ADX(14) for regime detection
    # +DM and -DM
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smoothed +DM, -DM, TR
    tr_rma = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_dm_rma = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_rma = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # +DI and -DI
    plus_di = 100 * plus_dm_rma / tr_rma
    minus_di = 100 * minus_dm_rma / tr_rma
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = np.full(n, np.nan)
    lowest_since_entry = np.full(n, np.nan)
    prev_bull_power = np.full(n, np.nan)
    prev_bear_power = np.full(n, np.nan)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx[i]) or np.isnan(atr[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            if i > 0:
                prev_bull_power[i] = prev_bull_power[i-1]
                prev_bear_power[i] = prev_bear_power[i-1]
            continue
        
        # Store previous values for power change detection
        if i > 0:
            prev_bull_power[i] = bull_power[i-1]
            prev_bear_power[i] = bear_power[i-1]
        else:
            prev_bull_power[i] = bull_power[i]
            prev_bear_power[i] = bear_power[i]
        
        if position == 0:
            # Regime-based entry logic
            if adx[i] > 25:  # Strong trend - trend following
                # LONG: Bull Power positive AND rising AND volume confirmation
                if (bull_power[i] > 0 and 
                    bull_power[i] > prev_bull_power[i] and 
                    volume[i] > 1.5 * avg_volume[i]):
                    signals[i] = 0.25
                    position = 1
                    highest_since_entry[i] = high[i]
                # SHORT: Bear Power negative AND falling AND volume confirmation
                elif (bear_power[i] < 0 and 
                      bear_power[i] < prev_bear_power[i] and 
                      volume[i] > 1.5 * avg_volume[i]):
                    signals[i] = -0.25
                    position = -1
                    lowest_since_entry[i] = low[i]
                else:
                    signals[i] = 0.0
            else:  # Ranging market (ADX < 25) - mean reversion at extremes
                # LONG: Bear Power deeply negative AND turning up AND volume confirmation
                if (bear_power[i] < -0.5 * atr[i] and 
                    bear_power[i] > prev_bear_power[i] and  # Turning up from extreme
                    volume[i] > 1.5 * avg_volume[i]):
                    signals[i] = 0.25
                    position = 1
                    highest_since_entry[i] = high[i]
                # SHORT: Bull Power deeply positive AND turning down AND volume confirmation
                elif (bull_power[i] > 0.5 * atr[i] and 
                      bull_power[i] < prev_bull_power[i] and  # Turning down from extreme
                      volume[i] > 1.5 * avg_volume[i]):
                    signals[i] = -0.25
                    position = -1
                    lowest_since_entry[i] = low[i]
                else:
                    signals[i] = 0.0
        elif position == 1:
            # Update highest high since entry
            highest_since_entry[i] = max(highest_since_entry[i-1], high[i])
            # EXIT LONG: trailing stop hit (2.0x ATR)
            trailing_stop = close[i] < (highest_since_entry[i] - 2.0 * atr[i])
            if trailing_stop:
                signals[i] = 0.0
                position = 0
                highest_since_entry[i] = np.nan
            else:
                signals[i] = 0.25
                if i > 0:
                    highest_since_entry[i] = highest_since_entry[i-1]
        elif position == -1:
            # Update lowest low since entry
            lowest_since_entry[i] = min(lowest_since_entry[i-1], low[i])
            # EXIT SHORT: trailing stop hit (2.0x ATR)
            trailing_stop = close[i] > (lowest_since_entry[i] + 2.0 * atr[i])
            if trailing_stop:
                signals[i] = 0.0
                position = 0
                lowest_since_entry[i] = np.nan
            else:
                signals[i] = -0.25
                if i > 0:
                    lowest_since_entry[i] = lowest_since_entry[i-1]
    
    return signals