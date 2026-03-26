#!/usr/bin/env python3
"""
Experiment #003: 4h Camarilla Pivot STRICT - Institutional Order Zones

HYPOTHESIS: Institutional order zones at Camarilla pivots work in ALL markets.
- Bull: long at S3/S4 with volume confirmation
- Bear: short at R3/R4 with volume confirmation
- Range: mean-revert between pivots

KEY FIX FROM #015 FAILURE:
- #015 got 1550 trades (too loose). This version uses STRICT confluence:
  ALL conditions required: pivot touch (<0.3 ATR) + volume spike (>2x) + 
  chop trending (<50) + trend bias (1d HMA). No vol-only or EMA-only entries.
- 24-bar cooldown between trades to prevent overtrading
- Target: 60-100 total trades over 4 years (proven pattern)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_camarilla_strict_vol_cooldown_v1"
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
    """Choppiness Index - CHOP < 50 = trending (allow trades)"""
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
    """Camarilla pivot levels from previous day"""
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
    
    # Load 1d data for Camarilla pivots and trend (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate Camarilla pivots from 1d
    cam_pivots = calculate_camarilla_pivots(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values
    )
    
    # Align pivots to 4h (with shift(1) for no look-ahead)
    s3_aligned = align_htf_to_ltf(prices, df_1d, cam_pivots['s3'])
    s4_aligned = align_htf_to_ltf(prices, df_1d, cam_pivots['s4'])
    r3_aligned = align_htf_to_ltf(prices, df_1d, cam_pivots['r3'])
    r4_aligned = align_htf_to_ltf(prices, df_1d, cam_pivots['r4'])
    
    # 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
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
    bars_since_exit = 999  # Cooldown counter
    
    warmup = 80  # Need enough bars for all indicators
    
    for i in range(warmup, n):
        bars_since_exit += 1
        
        # === CHECK ALL INDICATORS AVAILABLE ===
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(chop_14[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(s3_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME: Choppiness must indicate trending ===
        chop = chop_14[i]
        is_trending = chop < 50.0  # STRICT: only allow in clear trends
        
        # === TREND BIAS: Price above/below 1d HMA ===
        price_above_1d_hma = not np.isnan(hma_1d_aligned[i]) and close[i] > hma_1d_aligned[i]
        
        # === VOLUME CONFIRMATION: Must have spike ===
        vol_spike = vol_ratio[i] > 2.0  # STRICT: 2x instead of 1.5x
        
        # === PIVOT LEVELS ===
        s3 = s3_aligned[i]
        s4 = s4_aligned[i]
        r3 = r3_aligned[i]
        r4 = r4_aligned[i]
        
        # === ENTRY LOGIC - ALL conditions required ===
        desired_signal = 0.0
        
        if bars_since_exit >= 24:  # Cooldown: 24 bars (4 days) between trades
            if is_trending and vol_spike:  # Both required
                # LONG: Price at S3/S4 support + bullish bias
                if price_above_1d_hma:
                    # S3 touch (within 0.3 ATR)
                    if not np.isnan(s3):
                        dist_s3 = (close[i] - s3) / atr_14[i]
                        if dist_s3 > -0.3 and dist_s3 < 0.3:
                            desired_signal = SIZE
                    
                    # S4 touch (within 0.3 ATR)
                    if not np.isnan(s4) and desired_signal == 0.0:
                        dist_s4 = (close[i] - s4) / atr_14[i]
                        if dist_s4 > -0.3 and dist_s4 < 0.3:
                            desired_signal = SIZE
                
                # SHORT: Price at R3/R4 resistance + bearish bias
                if not price_above_1d_hma:
                    # R3 touch (within 0.3 ATR)
                    if not np.isnan(r3):
                        dist_r3 = (r3 - close[i]) / atr_14[i]
                        if dist_r3 > -0.3 and dist_r3 < 0.3:
                            desired_signal = -SIZE
                    
                    # R4 touch (within 0.3 ATR)
                    if not np.isnan(r4) and desired_signal == 0.0:
                        dist_r4 = (r4 - close[i]) / atr_14[i]
                        if dist_r4 > -0.3 and dist_r4 < 0.3:
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
            bars_since_exit = 0
        
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
            bars_since_exit = 0
        
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
                bars_since_exit = 0
        else:
            if in_position:
                # Check if we should stay in position
                pass
            else:
                bars_since_exit = 999
        
        signals[i] = desired_signal if in_position or desired_signal != 0 else 0.0
    
    return signals