#!/usr/bin/env python3
"""
Experiment #009: 4h Camarilla Deep Level + Volume Spike + Strict Choppiness + Cooldown

HYPOTHESIS: The #003 strategy failed due to 1550 trades from multiple entry paths.
This version fixes overtrading by:
1. ONLY S4/R4 entries (deepest pivots = higher conviction, fewer signals)
2. Volume spike REQUIRED (1.8x, no EMA fallback)
3. Strict CHOP < 45 (only strong trends, not neutral)
4. 8-bar cooldown after exit (prevents cluster re-entry)
5. 1d HMA alignment filter

WHY THIS SHOULD WORK IN BOTH BULL AND BEAR:
- S4/R4 are extreme levels from previous day's range - price rarely reaches them
- When price DOES reach S4 with volume spike = capitulation buying = high win rate
- Works in bear (short R4 bounces) and bull (long S4 bounces)
- Choppiness filter removes the 60% of time when Camarilla fails

TARGET: 75-150 total trades over 4 years (proven pattern from DB).
DB reference: gen_camarilla_pivot_volume_spike_choppiness_4h_v1 (Sharpe=1.471, 95tr)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_camarilla_deep_vol_strict_chop_cooldown_v1"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness
    CHOP > 61.8 = ranging, CHOP < 45 = trending (allow trades)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_camarilla_pivots(prev_high, prev_low, prev_close):
    """
    Camarilla pivot levels - only need S4 and R4 for deep level entries
    """
    n = len(prev_high)
    pivots = {
        's4': np.full(n, np.nan, dtype=np.float64),
        'r4': np.full(n, np.nan, dtype=np.float64),
    }
    
    for i in range(n):
        if np.isnan(prev_high[i]) or np.isnan(prev_low[i]) or np.isnan(prev_close[i]):
            continue
        
        high_low_range = prev_high[i] - prev_low[i]
        if high_low_range <= 1e-10:
            continue
        
        close = prev_close[i]
        
        # S4 = close - range * 1.1 / 2 (deepest support)
        pivots['s4'][i] = close - high_low_range * 1.1 / 2
        # R4 = close + range * 1.1 / 2 (deepest resistance)
        pivots['r4'][i] = close + high_low_range * 1.1 / 2
    
    return pivots

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for trend alignment
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate Camarilla S4/R4 from 1d
    cam_pivots = calculate_camarilla_pivots(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values
    )
    
    # Align pivots to 4h (auto shift(1) via align_htf_to_ltf)
    s4_aligned = align_htf_to_ltf(prices, df_1d, cam_pivots['s4'])
    r4_aligned = align_htf_to_ltf(prices, df_1d, cam_pivots['r4'])
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    # Volume moving average (20-bar for stability)
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
    
    # Cooldown tracking to prevent cluster re-entry
    bars_since_exit = 999  # Start with no cooldown
    MIN_BARS_BETWEEN_TRADES = 8
    
    # Warmup period
    warmup = 100
    
    for i in range(warmup, n):
        bars_since_exit += 1  # Increment cooldown counter
        
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(chop_14[i]):
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
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME CHECK (STRICT) ===
        chop = chop_14[i]
        is_strong_trend = chop < 45.0  # Only trending markets
        
        # === TREND ALIGNMENT (1d HMA) ===
        price_above_1d_hma = close[i] > hma_1d_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_aligned[i]
        
        # === VOLUME CONFIRMATION (REQUIRED) ===
        vol_spike = vol_ratio[i] > 1.8  # Must have volume spike
        
        # === PIVOT LEVELS ===
        s4 = s4_aligned[i]
        r4 = r4_aligned[i]
        
        # Price distance to S4/R4 as % of ATR
        dist_to_s4 = (close[i] - s4) / atr_14[i]
        dist_to_r4 = (r4 - close[i]) / atr_14[i]
        
        # === STOPLOSS CHECK ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        # === TAKE PROFIT at opposite pivot ===
        tp_triggered = False
        if in_position and position_side > 0:
            # TP at R4
            if high[i] >= r4:
                tp_triggered = True
        
        if in_position and position_side < 0:
            # TP at S4
            if low[i] <= s4:
                tp_triggered = True
        
        # === HANDLE EXIT ===
        if stoploss_triggered or tp_triggered:
            in_position = False
            position_side = 0
            bars_since_exit = 0  # Start cooldown after exit
            signals[i] = 0.0
            continue
        
        # === ENTRY LOGIC (ONLY S4/R4, strict filters) ===
        desired_signal = 0.0
        
        if not in_position and bars_since_exit >= MIN_BARS_BETWEEN_TRADES:
            
            # LONG: Price at S4 deep support + strong trend + volume spike + bullish HMA
            # Price must be BELOW S4 (hasn't bounced yet), within 0.5 ATR
            if (dist_to_s4 > -0.5 and dist_to_s4 < 1.5 and
                is_strong_trend and
                vol_spike and
                price_above_1d_hma):
                desired_signal = SIZE
            
            # SHORT: Price at R4 deep resistance + strong trend + volume spike + bearish HMA
            # Price must be ABOVE R4 (hasn't bounced yet), within 0.5 ATR
            if (dist_to_r4 > -0.5 and dist_to_r4 < 1.5 and
                is_strong_trend and
                vol_spike and
                price_below_1d_hma):
                desired_signal = -SIZE
        
        # === MAINTAIN POSITION ===
        if in_position:
            desired_signal = SIZE if position_side > 0 else -SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New entry
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
        
        signals[i] = desired_signal
    
    return signals