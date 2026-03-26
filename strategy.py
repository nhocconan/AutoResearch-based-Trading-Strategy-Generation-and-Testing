#!/usr/bin/env python3
"""
Experiment #005: 12h Donchian + Williams %R + Volume

HYPOTHESIS: Price channel breakouts (Donchian) identify structural shifts.
Williams %R catches reversals at extremes (works in bull rallies and bear bounces).
Volume confirms institutional involvement at key levels.
HTF (1d) SMA200 filters entries to trend direction only.

WHY THIS SHOULD WORK IN BOTH BULL AND BEAR:
- Donchian breakout: works in all markets (bull breakout, bear breakdown, range)
- Williams %R extremes: -80 = oversold (bull bounces), -20 = overbought (bear rallies)
- 1d SMA200 filter: ensures we're trading WITH the larger trend
- ATR stoploss: adapts to volatility in both high-squeeze (2021-2022) and low-vol (2025)

TARGET: 60-120 total trades over 4 years (15-30/year on 12h).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_williams_vol_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_williams_r(high, low, close, period=14):
    """Williams %R - momentum oscillator for overbought/oversold"""
    n = len(close)
    result = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        window_high = np.max(high[i - period + 1:i + 1])
        window_low = np.min(low[i - period + 1:i + 1])
        price_range = window_high - window_low
        
        if price_range > 1e-10:
            result[i] = -100.0 * (window_high - close[i]) / price_range
    
    return result

def calculate_donchian(high, low, period=20):
    """Donchian Channel - returns upper (highest) and lower (lowest) bands"""
    n = len(high)
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    middle = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
        middle[i] = (upper[i] + lower[i]) / 2.0
    
    return upper, lower, middle

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

def calculate_ema(close, span):
    """Exponential Moving Average"""
    return pd.Series(close).ewm(span=span, min_periods=span, adjust=False).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1d data for HTF trend filter (SMA200)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d SMA200 for trend filter
    sma_200_1d = pd.Series(df_1d['close']).rolling(window=200, min_periods=200).mean().values
    sma_200_aligned = align_htf_to_ltf(prices, df_1d, sma_200_1d)
    
    # Calculate 12h indicators
    williams_r = calculate_williams_r(high, low, close, period=14)
    donch_upper, donch_lower, donch_mid = calculate_donchian(high, low, period=20)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # EMA for momentum direction
    ema_8 = calculate_ema(close, 8)
    ema_21 = calculate_ema(close, 21)
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup: need ~200 bars for SMA200 alignment
    warmup = 250
    
    for i in range(warmup, n):
        # Skip if ATR not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # Skip if Williams %R not ready
        if np.isnan(williams_r[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # Skip if Donchian not ready
        if np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND FILTER (1d SMA200) ===
        price_above_sma200 = close[i] > sma_200_aligned[i] if not np.isnan(sma_200_aligned[i]) else True
        price_below_sma200 = close[i] < sma_200_aligned[i] if not np.isnan(sma_200_aligned[i]) else False
        
        # === WILLIAMS %R EXTREMES ===
        williams = williams_r[i]
        is_oversold = williams < -80  # Bullish reversal zone
        is_overbought = williams > -20  # Bearish reversal zone
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === DONCHIAN BREAKOUT CHECK ===
        donch_upper_val = donch_upper[i]
        donch_lower_val = donch_lower[i]
        donch_mid_val = donch_mid[i]
        
        # Price near upper band (breakout/resistance)
        dist_to_upper = (donch_upper_val - close[i]) / atr_14[i] if atr_14[i] > 0 else 999
        # Price near lower band (breakout/support)
        dist_to_lower = (close[i] - donch_lower_val) / atr_14[i] if atr_14[i] > 0 else 999
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: Price at lower Donchian band + oversold + above SMA200
        # This catches bounces in uptrends
        if is_oversold and price_above_sma200:
            # Price within 1.5 ATR of lower band
            if dist_to_lower < 1.5:
                if vol_spike:
                    desired_signal = SIZE
                elif close[i] > ema_21[i] if not np.isnan(ema_21[i]) else False:
                    desired_signal = SIZE
        
        # SHORT: Price at upper Donchian band + overbought + below SMA200
        # This catches reversals in downtrends
        if is_overbought and price_below_sma200:
            # Price within 1.5 ATR of upper band
            if dist_to_upper < 1.5:
                if vol_spike:
                    desired_signal = -SIZE
                elif close[i] < ema_21[i] if not np.isnan(ema_21[i]) else False:
                    desired_signal = -SIZE
        
        # === STOPLOSS CHECK (ATR-based) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TAKE PROFIT: opposite Donchian band ===
        tp_triggered = False
        if in_position and position_side > 0:
            # TP when reaching upper band
            if close[i] >= donch_upper_val:
                tp_triggered = True
        
        if in_position and position_side < 0:
            # TP when reaching lower band
            if close[i] <= donch_lower_val:
                tp_triggered = True
        
        if tp_triggered:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = desired_signal
    
    return signals