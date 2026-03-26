#!/usr/bin/env python3
"""
Experiment #007: 6h Elder Ray + Donchian Breakout + 1d EMA Trend

HYPOTHESIS: Elder Ray measures buying/selling pressure relative to EMA.
Combined with Donchian breakout, it captures institutional moves.
The 1d EMA cross provides trend bias to avoid counter-trend trades.

WHY THIS SHOULD WORK IN BOTH BULL AND BEAR:
- Bull markets: Bull Power > 0 + price above Donchian = strong longs
- Bear markets: Bear Power < 0 + price below Donchian = strong shorts  
- Range markets: Both powers near zero = no trades (avoid whipsaws)
- 6h captures daily institutional flow without 4h noise

KEY DESIGN:
1. Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
2. Donchian(20) breakout confirms momentum
3. 1d EMA cross for trend bias (only trade with trend)
4. ATR-based stoploss (2x ATR)
5. Discrete signal: 0.25

TARGET: 50-100 total trades over 4 years (~15-25/year on 6h)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_elder_ray_donchian_1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_ema(close, span):
    """Exponential Moving Average"""
    return pd.Series(close).ewm(span=span, min_periods=span, adjust=False).mean().values

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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - returns upper and lower bands"""
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_bull_bear_power(high, low, close, ema_period=13):
    """Elder Ray: Bull Power and Bear Power"""
    n = len(close)
    ema = calculate_ema(close, ema_period)
    
    bull_power = np.full(n, np.nan, dtype=np.float64)
    bear_power = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(ema_period, n):
        if not np.isnan(ema[i]):
            bull_power[i] = high[i] - ema[i]
            bear_power[i] = low[i] - ema[i]
    
    return bull_power, bear_power

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load 1d data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA for trend bias (align to 6h)
    ema_1d = calculate_ema(df_1d['close'].values, 21)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 1d EMA 50 for longer trend
    ema_1d_50 = calculate_ema(df_1d['close'].values, 50)
    ema_1d_50_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_50)
    
    # 1d close aligned
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, df_1d['close'].values)
    
    # === Calculate 6h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Elder Ray
    bull_power, bear_power = calculate_bull_bear_power(high, low, close, ema_period=13)
    
    # Donchian Channel
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # EMA on 6h for momentum confirmation
    ema_6h_8 = calculate_ema(close, 8)
    ema_6h_21 = calculate_ema(close, 21)
    
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    
    warmup = 80
    
    for i in range(warmup, n):
        # Check indicators ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 0:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(bull_power[i]) or np.isnan(bear_power[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(ema_1d_aligned[i]) or np.isnan(close_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === 1d TREND BIAS ===
        price_above_1d_ema = close[i] > ema_1d_aligned[i]
        price_above_1d_ema50 = close[i] > ema_1d_50_aligned[i] if not np.isnan(ema_1d_50_aligned[i]) else True
        ema21_above_ema50 = ema_1d_aligned[i] > ema_1d_50_aligned[i] if not np.isnan(ema_1d_50_aligned[i]) else True
        
        # Bullish trend: price above both EMAs
        is_bull_trend = price_above_1d_ema and (price_above_1d_ema50 or ema21_above_ema50)
        # Bearish trend: price below both EMAs
        is_bear_trend = not price_above_1d_ema and (not price_above_1d_ema50 or not ema21_above_ema50)
        
        # === 6h MOMENTUM ===
        bull_positive = bull_power[i] > 0
        bear_negative = bear_power[i] < 0
        
        # EMA cross on 6h
        ema_bullish_6h = ema_6h_8[i] > ema_6h_21[i] if not np.isnan(ema_6h_8[i]) else False
        ema_bearish_6h = ema_6h_8[i] < ema_6h_21[i] if not np.isnan(ema_6h_8[i]) else False
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_up = close[i] > donchian_upper[i]
        donchian_breakout_down = close[i] < donchian_lower[i]
        
        # Price mid-channel (not at extremes)
        channel_width = donchian_upper[i] - donchian_lower[i]
        price_in_upper_half = close[i] > (donchian_upper[i] + donchian_lower[i]) / 2
        price_in_lower_half = close[i] < (donchian_upper[i] + donchian_lower[i]) / 2
        
        # Volume confirmation
        vol_confirm = vol_ratio[i] > 1.3
        
        desired_signal = 0.0
        
        # === LONG ENTRY ===
        # Price breaks above Donchian + Bull Power > 0 + bullish 1d trend
        if is_bull_trend and donchian_breakout_up and bull_positive:
            # Additional confirmation: EMA cross bullish or volume spike
            if vol_confirm or ema_bullish_6h:
                desired_signal = SIZE
        
        # Alternative long: Pullback to middle of channel in uptrend
        if is_bull_trend and bull_positive and ema_bullish_6h:
            if price_in_upper_half and close[i] > ema_6h_21[i]:
                if vol_confirm:
                    desired_signal = SIZE
        
        # === SHORT ENTRY ===
        # Price breaks below Donchian + Bear Power < 0 + bearish 1d trend
        if is_bear_trend and donchian_breakout_down and bear_negative:
            # Additional confirmation: EMA cross bearish or volume spike
            if vol_confirm or ema_bearish_6h:
                desired_signal = -SIZE
        
        # Alternative short: Pullback to middle of channel in downtrend
        if is_bear_trend and bear_negative and ema_bearish_6h:
            if price_in_lower_half and close[i] < ema_6h_21[i]:
                if vol_confirm:
                    desired_signal = -SIZE
        
        # === STOPLOSS ===
        if in_position and position_side > 0:
            stop_price = entry_price - 2.0 * entry_atr
            if low[i] < stop_price:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            stop_price = entry_price + 2.0 * entry_atr
            if high[i] > stop_price:
                desired_signal = 0.0
        
        # === TAKE PROFIT ===
        # Exit when Elder Ray reverses
        if in_position and position_side > 0:
            if bear_power[i] < -atr_14[i] * 0.5:  # Bear power turns strongly negative
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            if bull_power[i] > atr_14[i] * 0.5:  # Bull power turns strongly positive
                desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
        
        signals[i] = desired_signal
    
    return signals