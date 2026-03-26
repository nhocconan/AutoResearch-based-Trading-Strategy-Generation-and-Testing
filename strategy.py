#!/usr/bin/env python3
"""
Experiment #011: 6h Camarilla Pivot + Volume + ADX Regime + 1d Trend Bias

HYPOTHESIS: Camarilla pivot levels from 1d represent institutional order zones where 
price reacts. On 6h, we wait for price to reach these zones WITH confirmation:
1. Volume spike (>1.8x average) = institutional participation
2. ADX > 22 = trending regime (pivots work better in trends than chop)
3. 1d HMA direction = trade only with HTF trend bias

WHY THIS WORKS IN BULL AND BEAR:
- Bull: long at S3/S4 when 1d HMA rising + volume confirms
- Bear: short at R3/R4 when 1d HMA falling + volume confirms
- ADX filter avoids choppy periods where pivots fail repeatedly

KEY DIFFERENCE FROM #007 (747 trades - overtrading):
- Stricter volume threshold (1.8x vs 1.5x)
- ADX regime filter (must be >22, not just any ADX)
- Require BOTH volume AND ADX for entry (not either/or)
- Discrete signals only (no small position changes)
- Target: 75-200 trades over 4 years (12-50/year)

DESIGN:
1. 1d Camarilla S3/S4/R3/R4 as structure
2. 1d HMA(21) for trend bias
3. 6h ADX(14) > 22 for regime
4. 6h Volume > 1.8x 20-avg for confirmation
5. Stoploss: 2.5 ATR trailing
6. Signal: 0.0, ±0.25, ±0.30 (discrete)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_camarilla_vol_adx_1d_v2"
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

def calculate_adx(high, low, close, period=14):
    """Average Directional Index"""
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    
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
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_di = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    dx = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if atr[i] > 1e-10:
            pdi = plus_di[i] / atr[i] * 100
            mdi = minus_di[i] / atr[i] * 100
            if pdi + mdi > 1e-10:
                dx[i] = abs(pdi - mdi) / (pdi + mdi) * 100
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

def calculate_camarilla_pivots(prev_high, prev_low, prev_close):
    """Camarilla pivot levels from previous bar"""
    n = len(prev_high)
    pivots = {
        's3': np.full(n, np.nan, dtype=np.float64),
        's4': np.full(n, np.nan, dtype=np.float64),
        'r3': np.full(n, np.nan, dtype=np.float64),
        'r4': np.full(n, np.nan, dtype=np.float64),
    }
    
    for i in range(n):
        if np.isnan(prev_high[i]) or np.isnan(prev_low[i]) or np.isnan(prev_close[i]):
            continue
        
        high_low_range = prev_high[i] - prev_low[i]
        if high_low_range <= 1e-10:
            continue
        
        close = prev_close[i]
        pivots['s3'][i] = close - high_low_range * 1.1 / 4
        pivots['s4'][i] = close - high_low_range * 1.1 / 2
        pivots['r3'][i] = close + high_low_range * 1.1 / 4
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
    
    # Calculate 1d indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 1d Camarilla pivots
    cam_pivots = calculate_camarilla_pivots(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values
    )
    
    # Align pivots to 6h
    s3_aligned = align_htf_to_ltf(prices, df_1d, cam_pivots['s3'])
    s4_aligned = align_htf_to_ltf(prices, df_1d, cam_pivots['s4'])
    r3_aligned = align_htf_to_ltf(prices, df_1d, cam_pivots['r3'])
    r4_aligned = align_htf_to_ltf(prices, df_1d, cam_pivots['r4'])
    
    # Calculate 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    adx_14 = calculate_adx(high, low, close, period=14)
    
    # Volume moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 1e-10, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.30
    SIZE_EXIT = 0.0
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup for all indicators
    warmup = 100
    
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
        
        if np.isnan(s3_aligned[i]) or np.isnan(r3_aligned[i]):
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
        
        # === REGIME CHECK (ADX) ===
        adx = adx_14[i]
        is_trending = adx > 22.0  # Only trade in trending markets
        
        # === TREND BIAS (1d HMA slope) ===
        hma_1d = hma_1d_aligned[i]
        hma_1d_prev = hma_1d_aligned[i-1] if i > 0 else np.nan
        
        if not np.isnan(hma_1d) and not np.isnan(hma_1d_prev):
            hma_rising = hma_1d > hma_1d_prev
            hma_falling = hma_1d < hma_1d_prev
        else:
            hma_rising = True
            hma_falling = False
        
        price_above_hma = close[i] > hma_1d if not np.isnan(hma_1d) else True
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.8  # Stricter than 1.5x
        
        # === CAMARILLA PIVOT LEVELS ===
        s3 = s3_aligned[i]
        s4 = s4_aligned[i]
        r3 = r3_aligned[i]
        r4 = r4_aligned[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if is_trending:
            # LONG: Near S3/S4 + 1d HMA rising + volume spike
            if hma_rising and price_above_hma:
                # At S3 support (within 1 ATR above)
                if not np.isnan(s3):
                    dist_s3 = (close[i] - s3) / atr_14[i]
                    if -0.5 < dist_s3 < 1.5:
                        if vol_spike:
                            desired_signal = SIZE_ENTRY
                
                # At S4 deeper support
                if desired_signal == 0.0 and not np.isnan(s4):
                    dist_s4 = (close[i] - s4) / atr_14[i]
                    if -0.5 < dist_s4 < 1.5:
                        if vol_spike:
                            desired_signal = SIZE_ENTRY
            
            # SHORT: Near R3/R4 + 1d HMA falling + volume spike
            if hma_falling and not price_above_hma:
                # At R3 resistance (within 1 ATR below)
                if not np.isnan(r3):
                    dist_r3 = (r3 - close[i]) / atr_14[i]
                    if -0.5 < dist_r3 < 1.5:
                        if vol_spike:
                            desired_signal = -SIZE_ENTRY
                
                # At R4 deeper resistance
                if desired_signal == 0.0 and not np.isnan(r4):
                    dist_r4 = (r4 - close[i]) / atr_14[i]
                    if -0.5 < dist_r4 < 1.5:
                        if vol_spike:
                            desired_signal = -SIZE_ENTRY
        
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
        
        # === TAKE PROFIT at opposite pivot ===
        tp_triggered = False
        if in_position and position_side > 0:
            if not np.isnan(r3) and high[i] >= r3:
                tp_triggered = True
            if not np.isnan(r4) and high[i] >= r4:
                tp_triggered = True
        
        if in_position and position_side < 0:
            if not np.isnan(s3) and low[i] <= s3:
                tp_triggered = True
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