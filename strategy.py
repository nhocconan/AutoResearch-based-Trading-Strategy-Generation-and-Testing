#!/usr/bin/env python3
"""
Experiment #022: Donchian Breakout + Volume Spike + Choppiness Regime (4h)

HYPOTHESIS: Donchian(20) breakout is the proven #1 pattern from 16K experiments.
Adding volume confirmation filters false breakouts.
Choppiness Index prevents trading in ranging markets.
HTF (1d) Donchian alignment ensures macro trend agrees with entry.

WHY IT WORKS IN BOTH BULL AND BEAR:
- Bull: Price breaks 20-bar high + above HMA trend + volume spike + chop<50
- Bear: Price breaks 20-bar low + below HMA trend + volume spike + chop<50
- Range (chop>61): No entries = no whipsaw losses

KEY INSIGHT from DB: "ONE strong signal (price channel breakout) + volume 
confirmation + regime filter" = the winning formula. Simple = fewer trades 
= less fee drag = better test generalization.

TARGET: 75-250 total trades over 4 years (19-62/year) - looser than Ichimoku
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_vol_chop_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(values, period):
    """Hull Moving Average for trend"""
    n = len(values)
    hma = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        half = period // 2
        sqrt_n = int(np.sqrt(period))
        
        # WMA half period
        wma_half = np.sum(values[i-half+1:i+1] * np.arange(1, half+1)) / np.sum(np.arange(1, half+1))
        # WMA full period  
        wma_full = np.sum(values[i-period+1:i+1] * np.arange(1, period+1)) / np.sum(np.arange(1, period+1))
        
        # Hull
        hma[i] = 2 * wma_half - wma_full
        # Second hull smoothing
        if i >= sqrt_n - 1:
            wma_hull = np.sum(hma[i-sqrt_n+1:i+1] * np.arange(1, sqrt_n+1)) / np.sum(np.arange(1, sqrt_n+1))
            hma[i] = wma_hull
    
    return hma

def calculate_donchian(high, low, period=20):
    """Donchian Channel - price channel breakout"""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - trending vs ranging"""
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        
        if hh != ll:
            # ATR sum
            atr_sum = np.sum(high[i-period+1:i+1] - low[i-period+1:i+1])
            chop[i] = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(period)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """ATR for stoploss sizing"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d Donchian for HTF trend
    upper_1d, lower_1d = calculate_donchian(df_1d['high'].values, df_1d['low'].values, period=20)
    close_1d = df_1d['close'].values
    
    # HTF: price near 1d highs = bull, near lows = bear
    htf_bull = close_1d > upper_1d * 0.99  # Within 1% of 1d high
    htf_bear = close_1d < lower_1d * 1.01  # Within 1% of 1d low
    htf_bull_aligned = align_htf_to_ltf(prices, df_1d, htf_bull.astype(float))
    htf_bear_aligned = align_htf_to_ltf(prices, df_1d, htf_bear.astype(float))
    
    # === Local 4h indicators ===
    upper_dc, lower_dc = calculate_donchian(high, low, period=20)
    
    # HMA for trend direction
    hma_48 = calculate_hma(close, 48)
    
    # Choppiness Index
    chop = calculate_choppiness(high, low, close, period=14)
    
    # ATR for stops
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    trailing_high = 0.0
    trailing_low = 0.0
    
    warmup = 100  # Donchian 20 + HMA 48 + volume 20
    
    for i in range(warmup, n):
        # NaN checks
        if np.isnan(atr_14[i]) or np.isnan(upper_dc[i]) or np.isnan(hma_48[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME CHECK ===
        # Chop < 50 = trending (good), chop > 61.8 = ranging (skip)
        chop_trending = chop[i] < 50
        chop_range = chop[i] > 61.8
        
        if chop_range:
            # Skip entries in ranging markets
            if not in_position:
                signals[i] = 0.0
                continue
        
        # === LOCAL TREND ===
        bull_trend = close[i] > hma_48[i]
        bear_trend = close[i] < hma_48[i]
        
        # === VOLUME CONFIRMATION ===
        vol_confirm = vol_ratio[i] > 1.5
        
        # === HTF TREND ===
        htf_bull_now = htf_bull_aligned[i] > 0.5 if not np.isnan(htf_bull_aligned[i]) else bull_trend
        htf_bear_now = htf_bear_aligned[i] > 0.5 if not np.isnan(htf_bear_aligned[i]) else bear_trend
        
        # === DONCHIAN BREAKOUT ===
        # Price breaks above 20-bar high
        bull_breakout = close[i] > upper_dc[i-1]
        # Price breaks below 20-bar low
        bear_breakout = close[i] < lower_dc[i-1]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG: Donchian breakout + local bull trend + volume + HTF bull or neutral
            if bull_breakout and bull_trend and vol_confirm:
                # Prefer HTF bull, but allow if not definitively bear
                if htf_bull_now or not htf_bear_now:
                    desired_signal = SIZE
            
            # SHORT: Donchian breakdown + local bear trend + volume + HTF bear or neutral
            if bear_breakout and bear_trend and vol_confirm:
                # Prefer HTF bear, but allow if not definitively bull
                if htf_bear_now or not htf_bull_now:
                    desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR trailing) ===
        if in_position:
            if position_side > 0:
                # Update trailing high
                if high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Trailing stop
                stop_price = trailing_high - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                
                # Exit if trend reverses
                if bear_trend and chop_trending:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
            
            elif position_side < 0:
                # Update trailing low
                if low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Trailing stop
                stop_price = trailing_low + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                
                # Exit if trend reverses
                if bull_trend and chop_trending:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
        
        # === MIN HOLD: 4 bars to avoid fee churn ===
        if in_position and (i - entry_bar) < 4:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                trailing_high = high[i]
                trailing_low = low[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals