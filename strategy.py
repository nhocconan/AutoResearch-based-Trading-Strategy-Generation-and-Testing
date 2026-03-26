#!/usr/bin/env python3
"""
Experiment #011: 6h Donchian Breakout + 1d Trend Bias + Volume Confirmation

HYPOTHESIS: Donchian channel breakouts (20-period high/low) capture genuine momentum 
when aligned with higher timeframe trend. 1d HMA provides weekly bias direction. 
Volume spike (1.3x 20-bar MA) confirms institutional participation, filtering false 
breakouts common in crypto.

WHY THIS WORKS IN BOTH BULL AND BEAR:
- Bull: Long breakout above Donchian high when 1d HMA bullish + volume spike
- Bear: Short breakout below Donchian low when 1d HMA bearish + volume spike
- ATR filter ensures breakout magnitude exceeds noise (>0.3 ATR beyond channel)
- 2.5 ATR trailing stop protects capital during reversals
- Minimum 3-bar hold prevents churn on false breakouts

TARGET: 75-200 total trades over 4 years (~19-50/year)
Timeframe: 6h

KEY DIFFERENCES from failed #009 (2443 trades):
1. Donchian breakout (rare event) vs Camarilla proximity (every bar)
2. AND logic: breakout + (trend OR volume) - stricter than OR-only
3. ATR magnitude filter: breakout must exceed channel by >0.3 ATR
4. Minimum 3-bar hold: no exit before bar 3
5. Trailing stoploss: 2.5 ATR from entry extreme
6. Discrete signals: only 0, +0.25, -0.25 (no gradual changes)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_donchian_1d_trend_vol_atr_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

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
    """Donchian Channel: upper = highest high, lower = lowest low over period"""
    n = len(high)
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Volume MA and ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_bar = 0
    bars_in_trade = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND BIAS (1d HMA) ===
        price_above_1d_hma = close[i] > hma_1d_aligned[i] if not np.isnan(hma_1d_aligned[i]) else True
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.3
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i]
        breakout_short = close[i] < donchian_lower[i]
        
        # === ATR MAGNITUDE FILTER ===
        # Breakout must exceed channel by >0.3 ATR (filters noise)
        long_valid = breakout_long and (close[i] - donchian_upper[i]) > 0.3 * atr_14[i]
        short_valid = breakout_short and (donchian_lower[i] - close[i]) > 0.3 * atr_14[i]
        
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
        
        # === TAKE PROFIT at opposite Donchian ===
        tp_triggered = False
        if in_position and position_side > 0 and not np.isnan(donchian_upper[i]):
            # Exit long at new Donchian high (momentum exhaustion)
            if high[i] >= donchian_upper[i] * 1.02:  # 2% beyond upper
                tp_triggered = True
        
        if in_position and position_side < 0 and not np.isnan(donchian_lower[i]):
            # Exit short at new Donchian low (momentum exhaustion)
            if low[i] <= donchian_lower[i] * 0.98:  # 2% beyond lower
                tp_triggered = True
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position or stoploss_triggered or tp_triggered:
            # Can consider new entry
            if stoploss_triggered or tp_triggered:
                # Just exited, need fresh signal
                pass
            
            # LONG: breakout + (bullish trend OR volume spike)
            if long_valid and (price_above_1d_hma or vol_spike):
                desired_signal = SIZE
            
            # SHORT: breakout + (bearish trend OR volume spike)
            if short_valid and ((not price_above_1d_hma) or vol_spike):
                desired_signal = -SIZE
        
        # === MINIMUM HOLDING PERIOD (3 bars) ===
        if in_position and bars_in_trade < 3 and not stoploss_triggered and not tp_triggered:
            # Cannot exit before bar 3 (keep same signal)
            desired_signal = signals[i-1] if i > 0 else 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side or stoploss_triggered or tp_triggered:
                # New position or flip or re-entry after exit
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                bars_in_trade = 0
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
            else:
                bars_in_trade += 1
        else:
            if in_position:
                in_position = False
                position_side = 0
                bars_in_trade = 0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = desired_signal
    
    return signals