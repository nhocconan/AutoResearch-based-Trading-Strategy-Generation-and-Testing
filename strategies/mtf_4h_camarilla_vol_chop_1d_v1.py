#!/usr/bin/env python3
"""
Experiment #003: 4h Camarilla Pivot + Volume Spike + Choppiness Regime

HYPOTHESIS: Camarilla pivot levels from 1d capture key support/resistance where 
institutions place orders. Volume spike confirms institutional involvement. 
Choppiness Index filters out choppy periods where pivots fail.

WHY THIS SHOULD WORK IN BOTH BULL AND BEAR:
- Camarilla works in ALL markets (bull, bear, range) because it's derived from 
  previous day's range, not trend direction
- Bear markets: short at R3/R4 with tight ATR stop
- Bull markets: long at S3/S4 with trailing stop
- Range markets: mean-revert between pivots

TARGET: 75-100 total trades over 4 years (proven pattern from DB).
DB reference: gen_camarilla_pivot_volume_spike_choppiness_4h_v1 (Sharpe=1.471)

KEY DESIGN:
1. Camarilla S3/S4/R3/R4 as entry zones
2. Volume spike confirmation (>1.5x 20-avg)
3. Choppiness filter (trend mode only)
4. ATR-based stoploss and take profit
5. 1d HMA for trend bias (only allow entries in trend direction)
6. Signal: 0.30 (discrete)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_camarilla_vol_chop_1d_v1"
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
    CHOP > 61.8 = ranging (no trades), CHOP < 50 = trending (allow trades)
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
    Camarilla pivot levels
    S1 = close + (high - low) * 1.1 / 12
    S2 = close + (high - low) * 1.1 / 6
    S3 = close + (high - low) * 1.1 / 4
    S4 = close + (high - low) * 1.1 / 2
    R1-R4 are symmetric below close
    """
    n = len(prev_high)
    pivots = {
        's1': np.full(n, np.nan, dtype=np.float64),
        's2': np.full(n, np.nan, dtype=np.float64),
        's3': np.full(n, np.nan, dtype=np.float64),
        's4': np.full(n, np.nan, dtype=np.float64),
        'r1': np.full(n, np.nan, dtype=np.float64),
        'r2': np.full(n, np.nan, dtype=np.float64),
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
        
        pivots['s1'][i] = close - high_low_range * 1.1 / 12
        pivots['s2'][i] = close - high_low_range * 1.1 / 6
        pivots['s3'][i] = close - high_low_range * 1.1 / 4
        pivots['s4'][i] = close - high_low_range * 1.1 / 2
        pivots['r1'][i] = close + high_low_range * 1.1 / 12
        pivots['r2'][i] = close + high_low_range * 1.1 / 6
        pivots['r3'][i] = close + high_low_range * 1.1 / 4
        pivots['r4'][i] = close + high_low_range * 1.1 / 2
    
    return pivots

def calculate_ema(close, span):
    """Exponential Moving Average"""
    return pd.Series(close).ewm(span=span, min_periods=span, adjust=False).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1d data for Camarilla pivots and trend
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate Camarilla pivots from 1d
    cam_pivots = calculate_camarilla_pivots(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values
    )
    
    # Align pivots to 4h
    s3_aligned = align_htf_to_ltf(prices, df_1d, cam_pivots['s3'])
    s4_aligned = align_htf_to_ltf(prices, df_1d, cam_pivots['s4'])
    r3_aligned = align_htf_to_ltf(prices, df_1d, cam_pivots['r3'])
    r4_aligned = align_htf_to_ltf(prices, df_1d, cam_pivots['r4'])
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    # Volume moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # EMA for trend confirmation
    ema_8 = calculate_ema(close, 8)
    ema_21 = calculate_ema(close, 21)
    
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
        
        if np.isnan(chop_14[i]):
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
        
        # === REGIME CHECK ===
        chop = chop_14[i]
        is_trending = chop < 55.0  # Allow entries in trending or neutral
        
        # === TREND BIAS (1d HMA + EMA cross) ===
        price_above_1d_hma = close[i] > hma_1d_aligned[i] if not np.isnan(hma_1d_aligned[i]) else True
        ema_bullish = ema_8[i] > ema_21[i] if not np.isnan(ema_8[i]) else True
        ema_bearish = ema_8[i] < ema_21[i] if not np.isnan(ema_8[i]) else False
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === CAMARILLA PIVOT LEVELS ===
        s3 = s3_aligned[i]
        s4 = s4_aligned[i]
        r3 = r3_aligned[i]
        r4 = r4_aligned[i]
        
        # Price distance to pivot levels (as % of ATR)
        if not np.isnan(s3) and atr_14[i] > 0:
            dist_to_s3 = (close[i] - s3) / atr_14[i]
            dist_to_s4 = (close[i] - s4) / atr_14[i] if not np.isnan(s4) else 999
            dist_to_r3 = (r3 - close[i]) / atr_14[i]
            dist_to_r4 = (r4 - close[i]) / atr_14[i] if not np.isnan(r4) else 999
        else:
            dist_to_s3 = 999
            dist_to_s4 = 999
            dist_to_r3 = 999
            dist_to_r4 = 999
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: Price near S3/S4 support + bullish bias + volume
        if is_trending:
            # At S3 support zone (within 0.5 ATR)
            if dist_to_s3 > -0.5 and dist_to_s3 < 2.0 and price_above_1d_hma:
                if vol_spike:
                    desired_signal = SIZE
                elif ema_bullish:
                    desired_signal = SIZE
            
            # At S4 deeper support
            if dist_to_s4 > -0.5 and dist_to_s4 < 2.0 and price_above_1d_hma:
                if vol_spike:
                    desired_signal = SIZE
                elif ema_bullish:
                    desired_signal = SIZE
        
        # SHORT: Price near R3/R4 resistance + bearish bias + volume
        if is_trending:
            # At R3 resistance zone (within 0.5 ATR)
            if dist_to_r3 > -0.5 and dist_to_r3 < 2.0 and not price_above_1d_hma:
                if vol_spike:
                    desired_signal = -SIZE
                elif ema_bearish:
                    desired_signal = -SIZE
            
            # At R4 deeper resistance
            if dist_to_r4 > -0.5 and dist_to_r4 < 2.0 and not price_above_1d_hma:
                if vol_spike:
                    desired_signal = -SIZE
                elif ema_bearish:
                    desired_signal = -SIZE
        
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
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TAKE PROFIT at opposite pivot ===
        tp_triggered = False
        if in_position and position_side > 0:
            # TP at R3/R4
            if not np.isnan(r3) and high[i] >= r3:
                tp_triggered = True
            if not np.isnan(r4) and high[i] >= r4:
                tp_triggered = True
        
        if in_position and position_side < 0:
            # TP at S3/S4
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
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
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