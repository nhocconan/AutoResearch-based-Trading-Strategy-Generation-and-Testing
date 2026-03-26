#!/usr/bin/env python3
"""
Experiment #007: 6h Elder Ray + Donchian Breakout + 1d Trend Filter

HYPOTHESIS: Elder Ray (Bull Power = High - EMA, Bear Power = Low - EMA) measures
institutional buying/selling pressure relative to fair value. Combined with 
Donchian breakout confirmation on 6h and 1d EMA trend filter, this captures 
the start of trends after consolidation.

WHY THIS SHOULD WORK IN BOTH BULL AND BEAR:
- Elder Ray is direction-agnostic (measures power relative to EMA)
- Bull Power > 0 = institutions buying above fair value
- Bear Power < 0 = institutions selling below fair value  
- Long: Bull Power > 0 + price breaks Donchian high + 1d trend up
- Short: Bear Power < 0 + price breaks Donchian low + 1d trend down
- ATR regime filter avoids whipsaws in low-vol periods

TARGET: 75-150 total trades over 4 years (12-37/year).
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
    """Donchian Channel - returns (upper, lower, mid)"""
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2.0
    return upper, lower, mid

def calculate_atr_percentile(atr, period=20):
    """ATR percentile for regime detection"""
    return pd.Series(atr).rolling(window=period, min_periods=period).apply(
        lambda x: (x[-1] - np.min(x)) / (np.max(x) - np.min(x) + 1e-10) * 100 
        if np.max(x) > np.min(x) else 50, raw=True
    ).values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA for trend filter
    ema_1d_raw = calculate_ema(df_1d['close'].values, span=21)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_raw)
    
    # 1d HMA for stronger trend detection
    def calculate_hma(close, period):
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
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # 6h indicators
    ema_6h = calculate_ema(close, span=21)
    atr_14 = calculate_atr(high, low, close, period=14)
    donch_upper, donch_lower, donch_mid = calculate_donchian(high, low, period=20)
    
    # Elder Ray calculation: power beyond EMA
    bull_power = high - ema_6h  # Buying pressure above EMA
    bear_power = low - ema_6h   # Selling pressure below EMA
    
    # Volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # ATR percentile for regime (volatility regime filter)
    atr_pct = calculate_atr_percentile(atr_14, period=20)
    
    signals = np.zeros(n)
    SIZE = 0.25  # Conservative sizing for 6h
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup for 1d alignment
    warmup = 80
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
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
        
        if np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND FILTER (1d) ===
        price_above_1d_hma = close[i] > hma_1d_aligned[i] if not np.isnan(hma_1d_aligned[i]) else True
        price_above_1d_ema = close[i] > ema_1d_aligned[i] if not np.isnan(ema_1d_aligned[i]) else True
        bull_trend = price_above_1d_hma and price_above_1d_ema
        bear_trend = not price_above_1d_hma and not price_above_1d_ema
        
        # === VOLATILITY REGIME FILTER ===
        # Only trade when ATR is not at extreme lows (avoid chop)
        atr_regime_ok = not (atr_pct[i] < 15)  # Skip if ATR at 30d low
        
        # === ELDER RAY SIGNALS ===
        bull_power_val = bull_power[i]
        bear_power_val = bear_power[i]
        
        # === DONCHIAN BREAKOUT ===
        donch_breakout_up = close[i] > donch_upper[i]
        donch_breakout_down = close[i] < donch_lower[i]
        
        # Previous bar was still inside channel (confirmation)
        prev_inside = donch_lower[i-1] < close[i-1] < donch_upper[i-1] if i > 0 else False
        
        # === VOLUME CONFIRMATION ===
        vol_confirm = vol_ratio[i] > 1.3
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if atr_regime_ok:
            # LONG: Bull Power positive + breaks above Donchian high + bull trend
            if bull_power_val > 0 and donch_breakout_up and bull_trend:
                if vol_confirm:
                    desired_signal = SIZE
                else:
                    # Still enter without volume, but smaller
                    desired_signal = SIZE * 0.5
            
            # SHORT: Bear Power negative + breaks below Donchian low + bear trend
            if bear_power_val < 0 and donch_breakout_down and bear_trend:
                if vol_confirm:
                    desired_signal = -SIZE
                else:
                    desired_signal = -SIZE * 0.5
        
        # === STOPLOSS CHECK ===
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
        
        # === EXIT ON TREND REVERSAL ===
        if in_position and position_side > 0:
            # Exit long if trend turns bearish
            if bear_trend and not bull_trend:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if trend turns bullish
            if bull_trend and not bear_trend:
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