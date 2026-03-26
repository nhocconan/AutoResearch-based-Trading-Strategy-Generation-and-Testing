#!/usr/bin/env python3
"""
Experiment #021: 1d Camarilla S3/R3 + Weekly Trend + Volume Spike

HYPOTHESIS: 1d Camarilla S3 (support) and R3 (resistance) with weekly trend 
confirmation and volume spike. 1d timeframe naturally limits trades to target range.

WHY THIS WORKS IN BOTH BULL AND BEAR:
- Bull: Long at S3 when weekly HMA trending up + volume spike
- Bear: Short at R3 when weekly HMA trending down + volume spike
- S3/R3 are the "sweet spots" - most reliable Camarilla levels

WHY 1d (not 4h):
- 1d = ~365 bars/year vs 4h = ~2190 bars/year (6x fewer bars = natural trade limit)
- DB: mtf_1d_kama_rsi_chop_regime_1w_v1 (74tr, Sharpe=1.31) = proven 1d edge

TARGET: 30-100 total trades over 4 years (7-25/year) - matches 1d HARD MAX of 150
KEY FIX from #016 (2443 trades - way overtraded):
- Tighter proximity: 0.3-1.0 ATR (not 0-2 ATR)
- Weekly HMA required (not optional)
- Volume spike 1.5x (not 1.3x)
- Minimum 15-bar hold to prevent signal churn
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_camarilla_s3r3_1w_trend_vol"
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

def calculate_camarilla_s3r3(prev_high, prev_low, prev_close):
    """
    Camarilla S3 and R3 levels (most reliable pivot levels)
    S3 = support below close (for longs)
    R3 = resistance above close (for shorts)
    """
    n = len(prev_high)
    pivots = {
        's3': np.full(n, np.nan, dtype=np.float64),
        'r3': np.full(n, np.nan, dtype=np.float64),
    }
    
    for i in range(n):
        if np.isnan(prev_high[i]) or np.isnan(prev_low[i]) or np.isnan(prev_close[i]):
            continue
        
        high_low_range = prev_high[i] - prev_low[i]
        if high_low_range <= 1e-10:
            continue
        
        close = prev_close[i]
        
        # S3 = support (below close)
        pivots['s3'][i] = close - high_low_range * 1.1 / 4
        # R3 = resistance (above close)
        pivots['r3'][i] = close + high_low_range * 1.1 / 4
    
    return pivots

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly HMA for trend direction
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate weekly HMA slope for trend strength
    hma_1w_slope_raw = np.zeros_like(hma_1w_raw)
    for i in range(3, len(hma_1w_raw)):
        if not np.isnan(hma_1w_raw[i]) and not np.isnan(hma_1w_raw[i-3]):
            hma_1w_slope_raw[i] = hma_1w_raw[i] - hma_1w_raw[i-3]
    hma_1w_slope_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_slope_raw)
    
    # Calculate Camarilla S3/R3 from daily data
    cam_pivots = calculate_camarilla_s3r3(
        prices["high"].values,
        prices["low"].values,
        close
    )
    s3 = cam_pivots['s3']
    r3 = cam_pivots['r3']
    
    # Calculate indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume MA and ratio
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
    bars_in_trade = 0
    
    warmup = 30
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(s3[i]) or np.isnan(r3[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === WEEKLY TREND CHECK (strict requirement) ===
        weekly_trend_bull = False
        weekly_trend_bear = False
        
        if not np.isnan(hma_1w_aligned[i]) and not np.isnan(hma_1w_slope_aligned[i]):
            # Weekly trend bullish: price above weekly HMA and HMA sloping up
            weekly_trend_bull = (close[i] > hma_1w_aligned[i] and hma_1w_slope_aligned[i] > 0)
            # Weekly trend bearish: price below weekly HMA and HMA sloping down
            weekly_trend_bear = (close[i] < hma_1w_aligned[i] and hma_1w_slope_aligned[i] < 0)
        
        # === VOLUME SPIKE (strict 1.5x) ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === CAMARILLA S3/R3 PROXIMITY (tight: 0.3-1.0 ATR) ===
        s3_dist = (close[i] - s3[i]) / atr_14[i]  # positive = price above S3
        r3_dist = (r3[i] - close[i]) / atr_14[i]  # positive = price below R3
        
        at_s3_zone = (0.3 <= s3_dist <= 1.0)  # Price 0.3-1.0 ATR above S3 (bouncing at support)
        at_r3_zone = (0.3 <= r3_dist <= 1.0)  # Price 0.3-1.0 ATR below R3 (bouncing at resistance)
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: At S3 + weekly bull trend + volume spike (ALL THREE required)
        if at_s3_zone and weekly_trend_bull and vol_spike:
            desired_signal = SIZE
        
        # SHORT: At R3 + weekly bear trend + volume spike (ALL THREE required)
        if at_r3_zone and weekly_trend_bear and vol_spike:
            desired_signal = -SIZE
        
        # === MINIMUM HOLD: 15 bars (prevent churn) ===
        if in_position and bars_in_trade < 15:
            # Keep position, don't exit early
            desired_signal = position_side * SIZE
        
        # === STOPLOSS CHECK (2.5 ATR) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered and bars_in_trade >= 15:
            desired_signal = 0.0
        
        # === TRAILING STOP (after 20 bars) ===
        if in_position and bars_in_trade >= 20:
            if position_side > 0:
                # Trail stop: lock in profits
                new_stop = high[i] - 2.0 * atr_14[i]
                stop_price = max(stop_price, new_stop)
            else:
                new_stop = low[i] + 2.0 * atr_14[i]
                stop_price = min(stop_price, new_stop)
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                bars_in_trade = 0
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
                bars_in_trade = 0
        
        signals[i] = desired_signal
        
        if in_position:
            bars_in_trade += 1
    
    return signals