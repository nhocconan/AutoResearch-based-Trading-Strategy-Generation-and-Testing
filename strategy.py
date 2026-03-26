#!/usr/bin/env python3
"""
Experiment #011: 6h Elder Ray + ADX Regime + 1d Trend Bias

HYPOTHESIS: Elder Ray measures institutional pressure (bull/bear power vs EMA).
Bull Power > 0 after being negative = smart money buying.
Bear Power < 0 after being positive = smart money selling.
Combined with ADX > 20 (trending) regime filter from 1d to avoid chop.
Works in both bull (follow bull power) and bear (follow bear power in downtrend).

Novel concept: Elder Ray has NOT been tried in any previous experiment.
TIMEFRAME: 6h primary
HTF: 1d for ADX regime and EMA trend
TARGET: 75-200 total trades over 4 years (12-37/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_elder_ray_adx_regime_v1"
timeframe = "6h"
leverage = 1.0

def calculate_ema(close, period):
    """Exponential Moving Average"""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - returns ADX series"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # True Range
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Directional Movement
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    # Smooth with EMA
    atr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # DI
    plus_di = np.zeros(n, dtype=np.float64)
    minus_di = np.zeros(n, dtype=np.float64)
    
    for i in range(period, n):
        if atr_smooth[i] > 0:
            plus_di[i] = 100 * plus_dm_smooth[i] / atr_smooth[i]
            minus_di[i] = 100 * minus_dm_smooth[i] / atr_smooth[i]
    
    # DX
    dx = np.zeros(n, dtype=np.float64)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    # ADX
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

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

def calculate_elder_ray(high, low, ema_values, period=13):
    """
    Elder Ray (Bull Power / Bear Power)
    Bull Power = High - EMA (buying pressure above EMA)
    Bear Power = Low - EMA (selling pressure below EMA)
    """
    n = len(high)
    bull_power = np.full(n, np.nan, dtype=np.float64)
    bear_power = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        if not np.isnan(ema_values[i]):
            bull_power[i] = high[i] - ema_values[i]
            bear_power[i] = low[i] - ema_values[i]
    
    return bull_power, bear_power

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA for trend direction
    ema_1d_raw = calculate_ema(df_1d['close'].values, period=21)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_raw)
    
    # 1d ADX for regime (trending vs ranging)
    adx_1d_raw = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d_raw)
    
    # === Local 6h indicators ===
    # EMA 13 for Elder Ray
    ema_13 = calculate_ema(close, period=13)
    
    # Elder Ray
    bull_power, bear_power = calculate_elder_ray(high, low, ema_13, period=13)
    
    # Local ADX for momentum confirmation
    adx_local = calculate_adx(high, low, close, period=14)
    
    # ATR for stops
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.25  # Conservative sizing
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    prev_bull_power = 0.0
    prev_bear_power = 0.0
    
    warmup = 60  # Need enough for ADX calculation
    
    for i in range(warmup, n):
        # Skip if key indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
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
        
        # Current values
        bull = bull_power[i] if not np.isnan(bull_power[i]) else 0.0
        bear = bear_power[i] if not np.isnan(bear_power[i]) else 0.0
        adx_1d = adx_1d_aligned[i] if not np.isnan(adx_1d_aligned[i]) else 0.0
        adx_6h = adx_local[i] if not np.isnan(adx_local[i]) else 0.0
        ema_21_local = ema_13[i] if not np.isnan(ema_13[i]) else close[i]  # Approximate
        
        # === TREND FILTER (1d EMA) ===
        # Only take longs when price above 1d EMA (bullish trend)
        # Only take shorts when price below 1d EMA (bearish trend)
        price_above_ema_1d = close[i] > ema_1d_aligned[i]
        price_below_ema_1d = close[i] < ema_1d_aligned[i]
        
        # === REGIME FILTER (1d ADX) ===
        # ADX > 20 = trending, ADX < 20 = ranging
        # Elder Ray works better in trending markets
        is_trending = adx_1d > 20
        
        # === ELDER RAY SIGNALS ===
        # Bull Power crosses from negative to positive = buying pressure
        # Bear Power crosses from positive to negative = selling pressure
        bull_turning_positive = (bull > 0) and (prev_bull_power <= 0)
        bear_turning_negative = (bear < 0) and (prev_bear_power >= 0)
        
        # Strong power: > 0.5 * ATR
        atr_val = atr_14[i]
        strong_bull = bull > 0.5 * atr_val
        strong_bear = bear < -0.5 * atr_val
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.2
        
        desired_signal = 0.0
        
        if not in_position:
            # === NEW LONG ENTRY ===
            # Bull power turning positive + price above 1d EMA + trending
            if bull_turning_positive and price_above_ema_1d and is_trending:
                if strong_bull or vol_spike:  # Require either strong power or volume
                    desired_signal = SIZE
            
            # === NEW SHORT ENTRY ===
            # Bear power turning negative + price below 1d EMA + trending
            if bear_turning_negative and price_below_ema_1d and is_trending:
                if strong_bear or vol_spike:
                    desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR) ===
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
        
        # === EXIT CONDITIONS ===
        exit_triggered = False
        
        if in_position and position_side > 0:
            # Exit long if:
            # - Bear power turns strongly negative (selling pressure)
            # - Price breaks below 1d EMA (trend change)
            # - ADX drops below 15 (regime change)
            if bear_turning_negative and strong_bear:
                exit_triggered = True
            if price_below_ema_1d:
                exit_triggered = True
            if adx_1d < 15:
                exit_triggered = True
        
        if in_position and position_side < 0:
            # Exit short if:
            # - Bull power turns strongly positive (buying pressure)
            # - Price breaks above 1d EMA (trend change)
            # - ADX drops below 15 (regime change)
            if bull_turning_positive and strong_bull:
                exit_triggered = True
            if price_above_ema_1d:
                exit_triggered = True
            if adx_1d < 15:
                exit_triggered = True
        
        if exit_triggered:
            desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
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
        
        # Store previous values for next iteration
        prev_bull_power = bull
        prev_bear_power = bear
        
        signals[i] = desired_signal
    
    return signals