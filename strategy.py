#!/usr/bin/env python3
"""
Experiment #004: 1d Donchian Breakout + TRIX Momentum + 1w Trend

HYPOTHESIS: 1d Donchian(20) breakouts mark institutional support/resistance
levels. Combined with TRIX momentum confirmation and 1w trend alignment,
this captures major trend moves while filtering noise. Using 1d primary
reduces trade frequency vs 4h/6h, minimizing fee drag. Works in both bull
(long breakouts above 1w SMA) and bear (short breakouts below 1w SMA).

TIMEFRAME: 1d primary
HTF: 1w for trend direction
TARGET: 50-150 total trades over 4 years (12-37/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_trix_1w_trend_v1"
timeframe = "1d"
leverage = 1.0

def calculate_trix(close, period=15):
    """TRIX - Triple EMA momentum oscillator"""
    n = len(close)
    if n < period * 3:
        return np.full(n, np.nan)
    
    ema1 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    ema2 = pd.Series(ema1).ewm(span=period, min_periods=period, adjust=False).mean().values
    ema3 = pd.Series(ema2).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    trix = np.full(n, np.nan, dtype=np.float64)
    for i in range(1, n):
        if not np.isnan(ema3[i]) and not np.isnan(ema3[i-1]) and ema3[i-1] != 0:
            trix[i] = ((ema3[i] - ema3[i-1]) / ema3[i-1]) * 100
    
    return trix

def calculate_donchian(high, low, period=20):
    """Donchian Channel - returns upper and lower bands"""
    n = len(high)
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    mid = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
        mid[i] = (upper[i] + lower[i]) / 2.0
    
    return upper, lower, mid

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
    
    # === Load HTF data ONCE ===
    df_1w = get_htf_data(prices, '1w')
    
    # 1w SMA for trend direction
    sma_1w = pd.Series(df_1w['close'].values).rolling(window=21, min_periods=21).mean().values
    sma_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_1w)
    
    # Calculate 1d indicators
    trix = calculate_trix(close, period=15)
    
    donch_upper, donch_lower, donch_mid = calculate_donchian(high, low, period=20)
    
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume ratio (20d MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
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
    
    warmup = 60
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(donch_upper[i]) or np.isnan(trix[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === 1w TREND (bullish if price above 1w SMA) ===
        trend_bullish = close[i] > sma_1w_aligned[i] if not np.isnan(sma_1w_aligned[i]) else True
        trend_bearish = close[i] < sma_1w_aligned[i] if not np.isnan(sma_1w_aligned[i]) else False
        
        # === TRIX MOMENTUM ===
        trix_val = trix[i]
        trix_cross_up = trix[i] > 0 and trix[i-1] <= 0 if i > 0 and not np.isnan(trix[i-1]) else False
        trix_cross_down = trix[i] < 0 and trix[i-1] >= 0 if i > 0 and not np.isnan(trix[i-1]) else False
        
        # === DONCHIAN BREAKOUT ===
        # Breakout = close breaks above/below yesterday's channel boundary
        price_above_upper = close[i] > donch_upper[i]
        price_below_lower = close[i] < donch_lower[i]
        
        # === VOLUME CONFIRMATION (optional but helps) ===
        vol_confirm = vol_ratio[i] > 1.2
        
        desired_signal = 0.0
        
        # === ENTRY LOGIC ===
        if not in_position:
            # LONG: breakout above upper band + bullish 1w trend + positive TRIX (or TRIX rising)
            if price_above_upper:
                if trend_bullish and (trix_val > 0 or trix_val > trix[i-2] if i > 1 and not np.isnan(trix[i-2]) else trix_val > -0.5):
                    desired_signal = SIZE
            
            # SHORT: breakout below lower band + bearish 1w trend + negative TRIX (or TRIX falling)
            if price_below_lower:
                if trend_bearish and (trix_val < 0 or trix_val < trix[i-2] if i > 1 and not np.isnan(trix[i-2]) else trix_val < 0.5):
                    desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR trailing stop) ===
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
        
        # === EXIT: TRIX reversal OR opposite channel touch ===
        exit_triggered = False
        
        if in_position and position_side > 0:
            # Exit long: TRIX crosses negative OR price hits lower band
            if trix_cross_down and trix_val < -0.1:
                exit_triggered = True
            if price_below_lower:
                exit_triggered = True
        
        if in_position and position_side < 0:
            # Exit short: TRIX crosses positive OR price hits upper band
            if trix_cross_up and trix_val > 0.1:
                exit_triggered = True
            if price_above_upper:
                exit_triggered = True
        
        if exit_triggered:
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