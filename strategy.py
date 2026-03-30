#!/usr/bin/env python3
"""
Experiment #004: 1d Donchian(55) Weekly Trend + Volume Spike + ATR Stop

HYPOTHESIS:
- 1d timeframe with 1w HTF reference should yield 30-100 trades over 4 years
- Use larger Donchian(55) = ~3 months to filter for major breakouts
- 1w HMA(21) as structural trend filter (weekly direction)
- Volume confirmation ensures institutional participation
- ATR(14) 2.5x stoploss for risk management

WHY IT SHOULD WORK IN BOTH BULL AND BEAR:
- Bull: Price > 1w HMA + breakout above 55d high = strong continuation
- Bear: Price < 1w HMA + breakdown below 55d low = strong short
- Range: 1w HMA flat + no breakout = no trades (filters sideways)
- Larger period (55 vs 20) = fewer but higher-quality signals

TARGET: 50-90 total trades over 4 years (12-22/year) on 1d
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian55_1w_hma_vol_atr_v1"
timeframe = "1d"
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

def calculate_hma(close, period=21):
    """Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    close_s = pd.Series(close)
    wma_half = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean().values
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean().values
    
    hull = 2 * wma_half - wma_full
    hma = pd.Series(hull).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean().values
    return hma

def calculate_donchian(high, low, period=55):
    """Donchian Channel - larger period for weekly structure"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_volume_ratio(volume, period=20):
    """Volume relative to moving average"""
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load 1w HTF data ONCE before loop ===
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly HMA(21) for structural trend
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Weekly EMA for additional confirmation
    ema_1w = pd.Series(df_1w['close'].values).ewm(span=13, min_periods=13, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # === Local 1d indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    donchian_up, donchian_lo = calculate_donchian(high, low, period=55)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.28
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    trailing_high = 0.0
    trailing_low = 0.0
    
    warmup = 150  # 55 for donchian + 20 for vol MA + HTF alignment
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_up[i]) or np.isnan(donchian_lo[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === Weekly trend filter (primary direction) ===
        weekly_trend_up = close[i] > hma_1w_aligned[i] and hma_1w_aligned[i] > ema_1w_aligned[i]
        weekly_trend_down = close[i] < hma_1w_aligned[i] and hma_1w_aligned[i] < ema_1w_aligned[i]
        
        # === Volume spike confirmation (2.0x threshold for quality) ===
        vol_spike = vol_ratio[i] > 2.0
        
        # === Donchian(55) breakout detection ===
        prev_up = donchian_up[i - 1] if i > 0 else donchian_up[i]
        prev_lo = donchian_lo[i - 1] if i > 0 else donchian_lo[i]
        
        # Breakout: close exceeds previous 55d high/low
        breakout_up = close[i] > prev_up
        breakout_down = close[i] < prev_lo
        
        # === Entry logic ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Weekly trend up + breakout above 55d high + volume spike ===
            if breakout_up and weekly_trend_up and vol_spike:
                desired_signal = SIZE
            
            # === SHORT: Weekly trend down + breakdown below 55d low + volume spike ===
            if breakout_down and weekly_trend_down and vol_spike:
                desired_signal = -SIZE
        
        # === STOPLOSS and position management ===
        if in_position:
            if position_side > 0:
                # Update trailing high
                if high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Trailing stop: 2.5 ATR from highest point
                stop_price = trailing_high - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Exit if weekly trend flips
                if not weekly_trend_up:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Update trailing low
                if low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Trailing stop: 2.5 ATR from lowest point
                stop_price = trailing_low + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Exit if weekly trend flips
                if not weekly_trend_down:
                    desired_signal = 0.0
        
        # === Minimum hold: 3 bars to avoid fee churn ===
        if in_position and (i - entry_bar) < 3:
            desired_signal = position_side * SIZE
        
        # === Update position ===
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