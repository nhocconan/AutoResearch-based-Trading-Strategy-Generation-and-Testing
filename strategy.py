#!/usr/bin/env python3
"""
Experiment #007: 6h Williams %R + 1d ATR Regime + Volume Spike

HYPOTHESIS: Volatility expansion (high ATR) at momentum extremes (%R near 0 or -100)
with volume confirmation captures institutional moves that start from oversold/overbought.
1d ATR regime ensures we only fade extremes when vol is elevated (catching reversals),
not in trending vol expansion (which would be countertrend).

WHY 6h: Slow enough for meaningful moves, fast enough for ~50 trades/4yr.
WHY IT WORKS IN BULL AND BEAR: Long when %R<10 + vol spike catches bear rallies.
Short when %R>-10 + vol spike catches bull selloffs. Symmetrical momentum fade.

TARGET: 75-150 total trades over 4 years = 19-37/year. HARD MAX: 300.
Signal size: 0.30.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_willr_atrregime_vol_1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_willr(high, low, close, period=14):
    """Williams %R - momentum oscillator"""
    n = len(close)
    result = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        highest = high[i - period + 1:i + 1].max()
        lowest = low[i - period + 1:i + 1].min()
        if highest != lowest:
            result[i] = -100.0 * (highest - close[i]) / (highest - lowest)
    
    return result

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d ATR for regime detection
    atr_1d_30 = calculate_atr(
        df_1d['high'].values, 
        df_1d['low'].values, 
        df_1d['close'].values, 
        period=30
    )
    atr_1d_7 = calculate_atr(
        df_1d['high'].values, 
        df_1d['low'].values, 
        df_1d['close'].values, 
        period=7
    )
    atr_1d_aligned_30 = align_htf_to_ltf(prices, df_1d, atr_1d_30)
    atr_1d_aligned_7 = align_htf_to_ltf(prices, df_1d, atr_1d_7)
    
    # 1d EMA200 for trend direction
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === Local 6h indicators ===
    willr_14 = calculate_willr(high, low, close, period=14)
    atr_6h = calculate_atr(high, low, close, period=14)
    
    # Volume ratio (20-bar lookback)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Day's range position (where is current close relative to day's range)
    day_high = pd.Series(high).rolling(window=4, min_periods=4).max().values
    day_low = pd.Series(low).rolling(window=4, min_periods=4).min().values
    day_range = day_high - day_low
    day_position = np.where(day_range > 0, (close - day_low) / day_range, 0.5)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    
    warmup = 250  # Need enough for EMA200 alignment buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(willr_14[i]) or np.isnan(atr_6h[i]) or atr_6h[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if HTF data not aligned
        if np.isnan(ema_1d_aligned[i]) or np.isnan(atr_1d_aligned_30[i]) or np.isnan(atr_1d_aligned_7[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1d EMA200) ===
        uptrend = close[i] > ema_1d_aligned[i]
        
        # === VOLATILITY REGIME (1d ATR ratio) ===
        # ATR expanding = more volatile = good for reversals
        # ATR contracting = less volatile = stay out
        vol_ratio_1d = atr_1d_aligned_7[i] / atr_1d_aligned_30[i] if atr_1d_aligned_30[i] > 0 else 1.0
        vol_expanding = vol_ratio_1d > 1.2  # 20% above 30d average
        
        # === MOMENTUM (Williams %R) ===
        # %R < -90 = deeply oversold
        # %R > -10 = deeply overbought
        oversold = willr_14[i] < -90
        overbought = willr_14[i] > -10
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === DAY'S RANGE POSITION ===
        # Near day's low = good for long entries
        # Near day's high = good for short entries
        near_day_low = day_position[i] < 0.25
        near_day_high = day_position[i] > 0.75
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Oversold + near day low + volume + uptrend + vol expanding ===
            if oversold and near_day_low and vol_spike and uptrend and vol_expanding:
                desired_signal = SIZE
            
            # === SHORT: Overbought + near day high + volume + downtrend + vol expanding ===
            if overbought and near_day_high and vol_spike and not uptrend and vol_expanding:
                desired_signal = -SIZE
        
        # === STOPLOSS (2.0 ATR trailing) ===
        if in_position:
            bars_held = i - entry_bar
            
            if position_side > 0:
                # Long stop
                if low[i] < entry_price - 2.0 * entry_atr:
                    desired_signal = 0.0
                # Take profit: %R back above -20 OR price near day high
                elif willr_14[i] > -20 or near_day_high:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Short stop
                if high[i] > entry_price + 2.0 * entry_atr:
                    desired_signal = 0.0
                # Take profit: %R back below -80 OR price near day low
                elif willr_14[i] < -80 or near_day_low:
                    desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_6h[i]
                entry_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals