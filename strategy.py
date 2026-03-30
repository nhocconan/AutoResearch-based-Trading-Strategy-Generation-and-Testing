#!/usr/bin/env python3
"""
Experiment #008: 12h Williams %R Extremes + ATR Volatility Expansion + SMA200 Trend

HYPOTHESIS: Williams %R extreme readings (<-80, >-20) mark reversals when 
combined with ATR expansion (volatility spike). The 1d SMA200 filters trend 
direction to avoid countertrend trades in strong trends.

WHY 12h: Slow enough for meaningful trades (~30-50/year), fast enough to 
capture reversals. 12h gives ~3x fewer trades than 4h = less fee drag.

WHY IT WORKS IN BULL AND BEAR:
- Bull: Buy oversold (%R<-80) at support bounces, ATR expansion confirms momentum
- Bear: Sell overbought (%R>-20) at resistance, ATR expansion confirms reversal

KEY INSIGHT: ATR expansion filters out low-volatility chop, which is the 
biggest killer of mean-reversion strategies. Only trade when volatility is expanding.

TARGET: 75-150 total trades over 4 years = 19-37/year. HARD MAX: 200.
Signal size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_willr_atr_expansion_1d_v2"
timeframe = "12h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_willr(high, low, close, period=14):
    """Williams %R"""
    n = len(close)
    willr = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        if highest_high != lowest_low:
            willr[i] = -100 * (highest_high - close[i]) / (highest_high - lowest_low)
    
    return willr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA200 for trend direction
    sma_200_1d = pd.Series(df_1d['close'].values).rolling(window=200, min_periods=200).mean().values
    sma_200_aligned = align_htf_to_ltf(prices, df_1d, sma_200_1d)
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    willr_14 = calculate_willr(high, low, close, period=14)
    
    # ATR expansion ratio (current ATR vs 30-bar MA)
    atr_ma30 = pd.Series(atr_14).rolling(window=30, min_periods=30).mean().values
    atr_ratio = atr_14 / np.where(atr_ma30 > 0, atr_ma30, 1)
    
    # Volume ratio
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 300  # Need 200 for SMA200 + buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(willr_14[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(sma_200_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND FILTER (1d SMA200) ===
        bull_trend = close[i] > sma_200_aligned[i]
        bear_trend = close[i] < sma_200_aligned[i]
        
        # === VOLATILITY EXPANSION FILTER ===
        # ATR ratio > 1.1 means volatility is expanding
        atr_expanding = atr_ratio[i] > 1.1
        
        # Volume confirmation
        vol_confirm = vol_ratio[i] > 1.2
        
        # === WILLIAMS %R EXTREME ZONES ===
        oversold = willr_14[i] < -80
        overbought = willr_14[i] > -20
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Oversold + ATR expanding + volume + bull trend ===
            if bull_trend and oversold and atr_expanding and vol_confirm:
                desired_signal = SIZE
            
            # === SHORT: Overbought + ATR expanding + volume + bear trend ===
            if bear_trend and overbought and atr_expanding and vol_confirm:
                desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR trailing) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            if low[i] < trailing_stop:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            if high[i] > trailing_stop:
                desired_signal = 0.0
        
        # === TIME EXIT: Hold for minimum 4 bars (2 days) to avoid fees ===
        bars_held = i - entry_bar
        if in_position and bars_held >= 4:
            # Take profit on Williams %R mean reversion
            if position_side > 0 and willr_14[i] > -50:
                desired_signal = 0.0
            if position_side < 0 and willr_14[i] < -50:
                desired_signal = 0.0
        
        # === MAX HOLD: Exit after 12 bars (6 days) ===
        if in_position and bars_held >= 12:
            desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                entry_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals