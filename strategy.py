#!/usr/bin/env python3
"""
Experiment #005: 12h Williams %R Mean Reversion + 1d Donchian Trend Confirmation

HYPOTHESIS: Williams %R reaching oversold (-80) or overbought (-20) extremes 
captures institutional order flow exhaustion points. Combined with 1d Donchian 
channel for trend confirmation, this catches reversals in both directions.

WHY THIS SHOULD WORK IN BOTH BULL AND BEAR:
- Williams %R is symmetric: extreme readings work whether trending up or down
- 1d Donchian filters direction: only short oversold bounces in downtrends
- Mean reversion at extremes is market-structure independent
- ATR-based stops ensure bounded risk regardless of market regime

TARGET: 75-150 total trades over 4 years (20-35/year).
Based on: Williams %R typically triggers 1-2x/month on crypto, filtered by trend.

KEY DESIGN:
1. Williams %R(14) < -80 for long, > -20 for short (extreme exhaustion)
2. 1d Donchian channel confirms trend direction
3. Volume spike confirmation (>1.5x 20-avg)
4. ATR-based stoploss (2x ATR) and take profit (3x ATR for R:R ~1.5)
5. Discrete signal: 0.25
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_willr_donchian_vol_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_williams_r(high, low, close, period=14):
    """Williams %R - momentum oscillator for mean reversion at extremes"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    willr = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        if highest_high - lowest_low > 1e-10:
            willr[i] = -100.0 * (highest_high - close[i]) / (highest_high - lowest_low)
    
    return willr

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
    """Donchian Channel - price channel breakout indicator"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan, dtype=np.float64)
    middle = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
        middle[i] = (upper[i] + lower[i]) / 2.0
    
    return upper, middle, lower

def calculate_ema(close, span):
    """Exponential Moving Average"""
    return pd.Series(close).ewm(span=span, min_periods=span, adjust=False).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1d data for Donchian channel (trend confirmation)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Donchian for trend
    donch_1d_up_raw, donch_1d_mid_raw, donch_1d_low_raw = calculate_donchian(
        df_1d['high'].values, df_1d['low'].values, period=20
    )
    
    # Align to 12h
    donch_up_aligned = align_htf_to_ltf(prices, df_1d, donch_1d_up_raw)
    donch_mid_aligned = align_htf_to_ltf(prices, df_1d, donch_1d_mid_raw)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_1d_low_raw)
    
    # Calculate 1d EMA for trend bias
    ema_1d_21_raw = calculate_ema(df_1d['close'].values, 21)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_21_raw)
    
    # Calculate 12h indicators
    willr_14 = calculate_williams_r(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    tp_price = 0.0
    
    # Warmup
    warmup = 80
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(willr_14[i]) or np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(donch_up_aligned[i]) or np.isnan(donch_low_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND CHECK (1d Donchian) ===
        price_above_donch_mid = close[i] > donch_mid_aligned[i] if not np.isnan(donch_mid_aligned[i]) else True
        price_near_donch_low = close[i] < donch_mid_aligned[i] if not np.isnan(donch_mid_aligned[i]) else False
        
        # 1d EMA trend
        ema_1d_trend_up = close[i] > ema_1d_aligned[i] if not np.isnan(ema_1d_aligned[i]) else True
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === WILLIAMS %R SIGNAL ===
        willr = willr_14[i]
        willr_oversold = willr < -80  # Extreme oversold - exhaustion
        willr_overbought = willr > -20  # Extreme overbought - exhaustion
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG: Williams %R oversold + price near 1d Donchian lower (support)
            # OR price above 1d EMA (bullish trend) + oversold reading
            if willr_oversold:
                if (price_above_donch_mid and ema_1d_trend_up) or vol_spike:
                    desired_signal = SIZE
            
            # SHORT: Williams %R overbought + price near 1d Donchian upper (resistance)
            # OR price below 1d EMA (bearish trend) + overbought reading
            if willr_overbought:
                if (price_near_donch_low and not ema_1d_trend_up) or vol_spike:
                    desired_signal = -SIZE
        
        # === STOPLOSS AND TAKE PROFIT ===
        stoploss_triggered = False
        tp_triggered = False
        
        if in_position and position_side > 0:
            # LONG: stop below entry - 2*ATR, TP at entry + 3*ATR
            if low[i] < stop_price:
                stoploss_triggered = True
            if high[i] >= tp_price:
                tp_triggered = True
        
        if in_position and position_side < 0:
            # SHORT: stop above entry + 2*ATR, TP at entry - 3*ATR
            if high[i] > stop_price:
                stoploss_triggered = True
            if low[i] <= tp_price:
                tp_triggered = True
        
        if stoploss_triggered or tp_triggered:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                if position_side > 0:
                    stop_price = entry_price - 2.0 * entry_atr
                    tp_price = entry_price + 3.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
                    tp_price = entry_price - 3.0 * entry_atr
        else:
            if in_position and (stoploss_triggered or tp_triggered):
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                tp_price = 0.0
        
        signals[i] = desired_signal
    
    return signals