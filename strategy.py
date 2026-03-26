#!/usr/bin/env python3
"""
Experiment #009: 4h Camarilla S4/R4 Extreme + Volume + 1d Trend + ADX

HYPOTHESIS: Only trade extreme Camarilla levels (S4/R4) where institutional orders cluster.
These are rare events (~2-3 per month per symbol) that generate 75-150 trades over 4 years.

WHY THIS WORKS IN BOTH BULL AND BEAR:
- S4/R4 are derived from previous day's range, not trend direction
- Bull market: long at S4 support with 1d HMA confirming uptrend
- Bear market: short at R4 resistance with 1d HMA confirming downtrend
- Volume spike (>2.0x) confirms institutional participation
- ADX > 25 filters out choppy periods where pivots fail
- 10-bar cooldown prevents re-entry whipsaw

TARGET: 75-150 total trades over 4 years (proven pattern from DB winner with 95 trades).
DB reference: gen_camarilla_pivot_volume_spike_choppiness_4h_v1 (Sharpe=1.471, 95tr)

KEY DESIGN:
1. ONLY S4/R4 levels (not S3/R3) - these are extreme and rare
2. Volume spike > 2.0x 20-avg (stricter than 1.5x)
3. 1d HMA for trend bias (mandatory alignment)
4. ADX > 25 for trending regime
5. 10-bar cooldown after exit
6. Signal: 0.30 (discrete)
7. ATR stoploss: 2.5x for wider stops on extreme levels
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_camarilla_s4r4_vol_adx_1d_v1"
timeframe = "4h"
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

def calculate_adx(high, low, close, period=14):
    """Average Directional Index"""
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, np.nan)
    
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(0, high[i] - high[i-1])
        else:
            plus_dm[i] = 0
        
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(0, low[i-1] - low[i])
        else:
            minus_dm[i] = 0
    
    plus_di = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_di = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di_pct = np.zeros(n, dtype=np.float64)
    minus_di_pct = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if atr[i] > 1e-10:
            plus_di_pct[i] = 100.0 * plus_di[i] / atr[i]
            minus_di_pct[i] = 100.0 * minus_di[i] / atr[i]
    
    dx = np.zeros(n, dtype=np.float64)
    for i in range(n):
        di_sum = plus_di_pct[i] + minus_di_pct[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di_pct[i] - minus_di_pct[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

def calculate_camarilla_pivots(prev_high, prev_low, prev_close):
    """
    Camarilla pivot levels - ONLY compute S4 and R4 (extreme levels)
    S4 = close - (high - low) * 1.1 / 2
    R4 = close + (high - low) * 1.1 / 2
    """
    n = len(prev_high)
    s4 = np.full(n, np.nan, dtype=np.float64)
    r4 = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(n):
        if np.isnan(prev_high[i]) or np.isnan(prev_low[i]) or np.isnan(prev_close[i]):
            continue
        
        high_low_range = prev_high[i] - prev_low[i]
        if high_low_range <= 1e-10:
            continue
        
        close = prev_close[i]
        s4[i] = close - high_low_range * 1.1 / 2
        r4[i] = close + high_low_range * 1.1 / 2
    
    return s4, r4

def calculate_ema(close, span):
    """Exponential Moving Average"""
    return pd.Series(close).ewm(span=span, min_periods=span, adjust=False).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate Camarilla S4/R4 from 1d
    s4_1d, r4_1d = calculate_camarilla_pivots(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values
    )
    
    # Align pivots to 4h
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    adx_14 = calculate_adx(high, low, close, period=14)
    
    # Volume moving average
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
    
    # Cooldown tracking (prevent re-entry whipsaw)
    bars_since_exit = 100  # Start high to allow initial entries
    COOLDOWN_BARS = 10
    
    # Warmup
    warmup = 60
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(adx_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(s4_aligned[i]) or np.isnan(r4_aligned[i]):
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
        
        # === REGIME CHECK: ADX > 25 (trending only) ===
        adx = adx_14[i]
        is_trending = adx > 25.0
        
        # === TREND BIAS (1d HMA) ===
        price_above_1d_hma = close[i] > hma_1d_aligned[i] if not np.isnan(hma_1d_aligned[i]) else True
        
        # === VOLUME CONFIRMATION (> 2.0x average) ===
        vol_spike = vol_ratio[i] > 2.0
        
        # === CAMARILLA S4/R4 LEVELS ===
        s4 = s4_aligned[i]
        r4 = r4_aligned[i]
        
        # Check if price is AT or BELOW S4 (long zone) or AT or ABOVE R4 (short zone)
        # Use 0.3 ATR tolerance for "at the level"
        atr = atr_14[i]
        at_s4 = (close[i] <= s4 + 0.3 * atr) and (close[i] >= s4 - 0.5 * atr)
        at_r4 = (close[i] >= r4 - 0.3 * atr) and (close[i] <= r4 + 0.5 * atr)
        
        # === ENTRY LOGIC (STRICT: all conditions must align) ===
        desired_signal = 0.0
        
        # LONG: At S4 + bullish 1d trend + volume spike + trending regime + cooldown ok
        if is_trending and price_above_1d_hma and vol_spike and at_s4 and bars_since_exit >= COOLDOWN_BARS:
            desired_signal = SIZE
        
        # SHORT: At R4 + bearish 1d trend + volume spike + trending regime + cooldown ok
        if is_trending and not price_above_1d_hma and vol_spike and at_r4 and bars_since_exit >= COOLDOWN_BARS:
            desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5x ATR for extreme levels) ===
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
        
        # === TAKE PROFIT at opposite pivot ===
        tp_triggered = False
        if in_position and position_side > 0:
            # TP at R4
            if not np.isnan(r4) and high[i] >= r4:
                tp_triggered = True
        
        if in_position and position_side < 0:
            # TP at S4
            if not np.isnan(s4) and low[i] <= s4:
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
                bars_since_exit = 0  # Reset cooldown on new entry
        else:
            if in_position:
                # Exited position
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                bars_since_exit = 0  # Start cooldown
            else:
                bars_since_exit += 1  # Increment cooldown if no position
        
        signals[i] = desired_signal
    
    return signals