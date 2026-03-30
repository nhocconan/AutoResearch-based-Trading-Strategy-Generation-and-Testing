#!/usr/bin/env python3
"""
Experiment #021: 4h TRIX + Donchian Breakout + Volume + 1d EMA Trend

HYPOTHESIS: TRIX momentum combined with Donchian(20) breakout catches
momentum shifts at key structural levels. Volume confirms institutional
activity. 1d EMA50 keeps us aligned with the broader trend.

WHY 4h: Proven timeframe from DB (multiple Sharpe 1.3+ strategies).
TRIX filters noise better than single/double EMA, catching genuine reversals.
Donchian(20) = 5 days = captures multi-day swings without overtrading.

TARGET: 75-200 total trades over 4 years (proven range). Signal size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_trix_donchian_vol_ema50_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_trix(close, period=9):
    """Triple EMA momentum oscillator"""
    n = len(close)
    if n < period * 3:
        return np.full(n, np.nan)
    
    ema1 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    ema2 = pd.Series(ema1).ewm(span=period, min_periods=period, adjust=False).mean().values
    ema3 = pd.Series(ema2).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # TRIX = rate of change of triple EMA
    trix = np.zeros(n)
    trix[period*3:] = (ema3[period*3:] - ema3[period*3 - period: -period]) / ema3[period*3 - period: -period] * 100
    
    return trix

def calculate_donchian(high, low, period=20):
    """Donchian Channel - returns upper, lower, mid"""
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2
    return upper, lower, mid

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
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
    
    # 1d EMA50 for trend (call ONCE, aligned)
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === 4h Indicators (pre-compute, no loop) ===
    trix = calculate_trix(close, period=9)
    donch_upper, donch_lower, donch_mid = calculate_donchian(high, low, period=20)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # TRIX signal line (9-period EMA of TRIX)
    trix_ema = pd.Series(trix).ewm(span=5, min_periods=5, adjust=False).mean().values
    
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_atr = 0.0
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 100  # TRIX needs ~27 bars, Donchian needs 20
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(trix[i]) or np.isnan(donch_lower[i]) or np.isnan(atr_14[i]) or atr_14[i] <= 0:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if HTF EMA not aligned
        if np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1d EMA50) ===
        price_above_1d_ema = close[i] > ema_1d_aligned[i]
        price_below_1d_ema = close[i] < ema_1d_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === TRIX MOMENTUM ===
        trix_turning_up = trix[i] > trix_ema[i] and trix[i-1] <= trix_ema[i-1]  # cross above signal
        trix_turning_down = trix[i] < trix_ema[i] and trix[i-1] >= trix_ema[i-1]  # cross below signal
        
        # Extreme TRIX for reversal (low = oversold, high = overbought)
        trix_oversold = trix[i] < -0.5
        trix_overbought = trix[i] > 0.5
        
        # === DONCHIAN BREAKOUT ===
        # Price breaks above 20-period high
        donch_breakout_up = close[i] > donch_upper[i]
        # Price breaks below 20-period low
        donch_breakout_down = close[i] < donch_lower[i]
        
        # Price at mid-channel (mean reversion target)
        at_mid_channel = abs(close[i] - donch_mid[i]) < donch_mid[i] * 0.01
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: TRIX turning up from oversold + Donchian breakout + trend alignment ===
            if price_above_1d_ema and (trix_turning_up or trix_oversold) and donch_breakout_up and vol_spike:
                desired_signal = SIZE
            
            # === SHORT: TRIX turning down from overbought + Donchian breakdown + trend alignment ===
            if price_below_1d_ema and (trix_turning_down or trix_overbought) and donch_breakout_down and vol_spike:
                desired_signal = -SIZE
        
        # === TRAILING STOPLOSS (2.5 ATR) ===
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
        
        # === TAKE PROFIT: price returns to mid-channel ===
        bars_held = i - entry_bar
        if in_position and bars_held >= 3:  # Hold at least 3 bars (12h)
            if position_side > 0 and at_mid_channel:
                desired_signal = 0.0
            if position_side < 0 and at_mid_channel:
                desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_atr = atr_14[i]
                entry_bar = i
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals