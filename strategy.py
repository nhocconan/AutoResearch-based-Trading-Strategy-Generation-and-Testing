#!/usr/bin/env python3
"""
Experiment #028: 4h TRIX + Volume Spike + Choppiness + ATR Channel

HYPOTHESIS: TRIX (triple smoothed momentum) provides cleaner signals than 
single/double EMA derivatives. By combining TRIX crossover with strict 
volume confirmation (1.8x) and Choppiness regime (<52 = trending), this 
captures medium-term momentum while avoiding range-bound whipsaws.

WHY 4h: Moderate frequency (target 75-150 trades/4yr), balances alpha vs fees.
TRIX period 15 catches medium-term swings, not noise.

WHY IT WORKS IN BULL AND BEAR:
- Long entries: TRIX crosses above signal line + price > SMA50 + volume spike
- Short entries: TRIX crosses below signal line + price < SMA50 + volume spike
- ATR-based stoploss adapts to volatility in both directions
- Symmetric logic handles both bull breakouts and bear breakdowns

TARGET: 75-150 total trades over 4 years. HARD MAX: 300.
Signal size: 0.30
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_trix_vol_chop_sma50_v1"
timeframe = "4h"
leverage = 1.0

def calculate_trix(prices, period=15):
    """Triple EMA derivative - momentum indicator with triple smoothing"""
    n = len(prices)
    if n < period * 3:
        return np.full(n, np.nan)
    
    # Triple EMA smoothing
    ema1 = pd.Series(prices).ewm(span=period, min_periods=period, adjust=False).mean().values
    ema2 = pd.Series(ema1).ewm(span=period, min_periods=period, adjust=False).mean().values
    ema3 = pd.Series(ema2).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # TRIX = rate of change of triple EMA
    trix = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if ema3[i - period] != 0:
            trix[i] = 100 * (ema3[i] - ema3[i - period]) / ema3[i - period]
    
    return trix

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

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - measures market choppiness (not direction)"""
    n = len(high)
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        tr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            if j > 0:
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]))
            else:
                tr = high[j] - low[j]
            tr_sum += tr
        
        if tr_sum > 0:
            hh = np.max(high[i - period + 1:i + 1])
            ll = np.min(low[i - period + 1:i + 1])
            range_hl = hh - ll
            
            if range_hl > 0:
                chop[i] = 100 * np.log10(tr_sum / range_hl) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA50 for trend direction
    sma_50_1d = pd.Series(df_1d['close'].values).rolling(window=50, min_periods=50).mean().values
    sma_50_aligned = align_htf_to_ltf(prices, df_1d, sma_50_1d)
    
    # Local 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # TRIX (period 15) + Signal line (period 9)
    trix = calculate_trix(close, period=15)
    signal = pd.Series(trix).ewm(span=9, min_periods=9, adjust=False).mean().values
    
    # Volume ratio (20-bar SMA)
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
    entry_bar = 0
    prev_trix = 0.0
    prev_signal = 0.0
    
    warmup = 150  # Need enough for TRIX triple smoothing + SMA50
    
    for i in range(warmup, n):
        # Check if indicators ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(trix[i]) or np.isnan(signal[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            prev_trix = 0.0
            prev_signal = 0.0
            continue
        
        if np.isnan(sma_50_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === REGIME CHECK (Choppiness Index) ===
        # Only trade when trending (CHOP < 52) - stricter than typical 61.8
        is_trending = chop[i] < 52.0
        
        # Skip new entries if too choppy
        if not in_position and not is_trending:
            signals[i] = 0.0
            prev_trix = trix[i]
            prev_signal = signal[i]
            continue
        
        # === TREND DIRECTION (1d SMA50) ===
        price_above_1d_sma = close[i] > sma_50_aligned[i]
        price_below_1d_sma = close[i] < sma_50_aligned[i]
        
        # === VOLUME CONFIRMATION (strict: 1.8x) ===
        vol_spike = vol_ratio[i] > 1.8
        
        # === TRIX CROSSOVER DETECTION ===
        trix_crossed_above = (prev_trix < prev_signal) and (trix[i] > signal[i])
        trix_crossed_below = (prev_trix > prev_signal) and (trix[i] < signal[i])
        
        # === ATR-BASED MOVEMENT FILTER ===
        # Price must move >0.8 ATR to confirm momentum
        price_change = abs(close[i] - close[i - 1]) if i > 0 else 0
        atr_movement = price_change / atr_14[i] if atr_14[i] > 0 else 0
        
        desired_signal = 0.0
        
        # === ENTRY LOGIC ===
        if not in_position:
            # === LONG: TRIX crosses above signal + price > SMA50 + volume ===
            if trix_crossed_above and price_above_1d_sma:
                if vol_spike or atr_movement > 0.8:  # Volume OR strong momentum
                    desired_signal = SIZE
            
            # === SHORT: TRIX crosses below signal + price < SMA50 + volume ===
            if trix_crossed_below and price_below_1d_sma:
                if vol_spike or atr_movement > 0.8:  # Volume OR strong momentum
                    desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR trailing) ===
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
        
        # === TIME-BASED EXIT (hold at least 8 bars = 1.3 days) ===
        bars_held = i - entry_bar
        
        if in_position and bars_held >= 8:
            # Exit on opposite TRIX crossover (momentum shift)
            if position_side > 0 and trix_crossed_below:
                desired_signal = 0.0
            if position_side < 0 and trix_crossed_above:
                desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
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
        
        # Update TRIX state for next iteration
        prev_trix = trix[i]
        prev_signal = signal[i]
        
        signals[i] = desired_signal
    
    return signals