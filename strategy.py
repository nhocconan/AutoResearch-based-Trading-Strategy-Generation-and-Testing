#!/usr/bin/env python3
"""
Experiment #004: 1d Williams %R Reversal + Donchian Confirmation

HYPOTHESIS: Williams %R extreme readings (<-80 or >-20) mark reversal points.
Combined with Donchian(20) channel touch for structural confirmation and 
1w HMA for trend direction, this captures major mean-reversion moves.
1d timeframe naturally limits trades to 50-150 total over 4 years.

WHY BOTH BULL AND BEAR: 
- Bull markets: %R oversold bounces + price above 1w HMA catch rallies
- Bear markets: %R overbought shorts + price below 1w HMA catch breakdowns
- Range markets: %R extremes at channel boundaries catch reversals

TIMEFRAME: 1d primary
HTF: 1w for trend direction (aligned once, used as filter)
TARGET: 50-150 total trades over 4 years (12.5-37.5/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_williams_donchian_1w_v1"
timeframe = "1d"
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

def calculate_williams_r(high, low, close, period=20):
    """Williams %R"""
    n = len(close)
    williams = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        if highest_high != lowest_low:
            williams[i] = -100 * (highest_high - close[i]) / (highest_high - lowest_low)
    
    return williams

def calculate_donchian(high, low, period=20):
    """Donchian Channel - returns upper and lower bands"""
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
    
    # === Load HTF data ONCE ===
    df_1w = get_htf_data(prices, '1w')
    
    # 1w HMA for trend direction
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Local indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    williams_r = calculate_williams_r(high, low, close, period=20)
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    
    # Volume MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Previous Donchian for breakout detection (shifted by 1 for no look-ahead)
    donch_upper_prev = np.roll(donch_upper, 1)
    donch_lower_prev = np.roll(donch_lower, 1)
    donch_upper_prev[0] = np.nan
    donch_lower_prev[0] = np.nan
    
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
    target_price = 0.0
    
    warmup = 60  # Need enough for 1w HMA alignment
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(williams_r[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND DIRECTION (1w HMA) ===
        trend_bull = close[i] > hma_1w_aligned[i]
        
        # === WILLIAMS %R EXTREMES ===
        williams_oversold = williams_r[i] < -80   # Strong oversold
        williams_overbought = williams_r[i] > -20  # Strong overbought
        
        # === DONCHIAN TOUCH (structural confirmation) ===
        near_lower = close[i] <= donch_lower[i] * 1.02  # Within 2% of lower band
        near_upper = close[i] >= donch_upper[i] * 0.98  # Within 2% of upper band
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.3
        
        # === ATR FILTER (enough volatility for 2.5 ATR stop) ===
        atr_ratio = atr_14[i] / close[i]
        has_volatility = atr_ratio >= 0.015  # Need ~1.5% daily ATR minimum
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG ENTRY: %R oversold + near lower band + bullish trend ===
            if williams_oversold and near_lower and trend_bull and has_volatility:
                desired_signal = SIZE
            
            # === SHORT ENTRY: %R overbought + near upper band + bearish trend ===
            if williams_overbought and near_upper and not trend_bull and has_volatility:
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                desired_signal = 0.0
        
        # === TARGET CHECK (3R profit) ===
        if in_position and target_price > 0:
            if position_side > 0 and high[i] >= target_price:
                # Take profit at 3R
                desired_signal = SIZE / 2