#!/usr/bin/env python3
"""
Experiment #022: 1d Donchian Breakout + HMA + Volume + Weekly Trend

HYPOTHESIS: Simple price-channel breakout works on 1d better than complex indicators.
- Bull: Price breaks 20d high + HMA trending up + volume spike + weekly bull
- Bear: Price breaks 20d low + HMA trending down + volume spike + weekly bear
- Range: No trade (avoid whipsaws in choppy markets)

WHY IT SHOULD WORK BOTH MARKETS:
- 2021-2022 bull: Breakouts fire, HMA confirms, big volume on breakouts
- 2022 crash: Short side works when price breaks 20d low with volume
- 2023-2024 range: Fewer signals (good!) but what fires tends to work
- 1d = ~250 bars/year, 20d Donchian = ~12-15 breakout signals/year max

TARGET: 50-80 total trades over 4 years (12-20/year) — LOW fee drag
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_hma_vol_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(data, period):
    """Hull Moving Average"""
    n = len(data)
    if n < period:
        return np.full(n, np.nan)
    
    # WMA half period
    half = int(period / 2)
    wma_half = pd.Series(data).rolling(window=half, min_periods=half).mean().values
    
    # WMA full period  
    wma_full = pd.Series(data).rolling(window=period, min_periods=period).mean().values
    
    # HMA = 2 * WMA(half) - WMA(full)
    hma = 2 * wma_half - wma_full
    
    return hma

def calculate_donchian(high, low, period=20):
    """Donchian Channel - use price position, not breakout trigger"""
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2
    return upper, lower, mid

def calculate_volume_ratio(volume, period=20):
    """Volume relative to 20d average"""
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly HMA for macro trend
    hma_21_1w = calculate_hma(df_1w['close'].values, 21)
    weekly_bull = hma_21_1w > pd.Series(hma_21_1w).shift(5).values
    weekly_bear = hma_21_1w < pd.Series(hma_21_1w).shift(5).values
    
    weekly_bull_aligned = align_htf_to_ltf(prices, df_1w, weekly_bull.astype(float))
    weekly_bear_aligned = align_htf_to_ltf(prices, df_1w, weekly_bear.astype(float))
    
    # === Local 1d indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # HMA for trend direction
    hma_21 = calculate_hma(close, 21)
    hma_trending_up = hma_21 > pd.Series(hma_21).shift(3).values
    hma_trending_down = hma_21 < pd.Series(hma_21).shift(3).values
    
    # Donchian channels
    donch_upper_20, donch_lower_20, _ = calculate_donchian(high, low, period=20)
    donch_upper_55, donch_lower_55, _ = calculate_donchian(high, low, period=55)
    
    # Volume ratio
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
    # Position signals
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    trailing_high = 0.0
    trailing_low = 0.0
    
    warmup = 100  # 55 for Donchian + 21 for HMA + buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_21[i]) or np.isnan(donch_upper_20[i]):
            signals[i] = 0.0
            continue
        
        # === WEEKLY TREND (HTF) ===
        is_weekly_bull = weekly_bull_aligned[i] > 0.5 if not np.isnan(weekly_bull_aligned[i]) else False
        is_weekly_bear = weekly_bear_aligned[i] > 0.5 if not np.isnan(weekly_bear_aligned[i]) else False
        
        # === ENTRY CONDITIONS ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG: Price breaks 20d high + HMA trending up + volume spike + weekly bull
            price_near_high = close[i] >= donch_upper_20[i] * 0.98  # Within 2% of high
            vol_spike = vol_ratio[i] > 1.6
            
            bull_conditions = (price_near_high and hma_trending_up[i] and 
                             vol_spike and (is_weekly_bull or not is_weekly_bear))
            
            if bull_conditions:
                desired_signal = SIZE
            
            # SHORT: Price breaks 20d low + HMA trending down + volume spike + weekly bear
            price_near_low = close[i] <= donch_lower_20[i] * 1.02  # Within 2% of low
            vol_spike_short = vol_ratio[i] > 1.6
            
            bear_conditions = (price_near_low and hma_trending_down[i] and 
                             vol_spike_short and (is_weekly_bear or not is_weekly_bull))
            
            if bear_conditions:
                desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR trailing stop) ===
        if in_position:
            if position_side > 0:
                # Update trailing high
                if i == entry_bar or high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Trailing stop
                stop_price = trailing_high - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                
                # Exit if HMA turns bearish
                if hma_trending_down[i]:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
            
            elif position_side < 0:
                # Update trailing low
                if i == entry_bar or low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Trailing stop
                stop_price = trailing_low + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                
                # Exit if HMA turns bullish
                if hma_trending_up[i]:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
        
        # === MINIMUM HOLD: 3 bars to avoid fee churn ===
        if in_position and (i - entry_bar) < 3:
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