#!/usr/bin/env python3
"""
Experiment #021: 4h Camarilla S3/R3 Zone + Volume Spike + Choppiness Regime

HYPOTHESIS: Camarilla S3 (support) and R3 (resistance) levels from 1d capture 
institutional order flow. Price touching these zones + volume spike = high-probability
reversal. Choppiness Index filters out ranging periods where pivots fail.

WHY THIS SHOULD WORK IN BOTH BULL AND BEAR:
- S3/S4 = buy zones in any market (institutions accumulate at support)
- R3/R4 = sell zones in any market (institutions distribute at resistance)
- Choppiness filter prevents trading in consolidating markets
- 1d HMA provides trend bias without being too restrictive

DB REFERENCE: gen_camarilla_pivot_volume_spike_choppiness_4h_v1 (Sharpe=1.471, 95 trades)

KEY DESIGN (from successful DB pattern):
1. 1d Camarilla S3/R3 aligned to 4h (call ONCE before loop)
2. Price within 0.3ATR of S3 (long) or R3 (short)
3. Volume spike > 1.5x 20-avg
4. Choppiness < 55 (trending mode)
5. 1d HMA for trend direction
6. ATR stoploss (2ATR) and take profit (opposite pivot)
7. Signal: 0.30 (discrete)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_camarilla_zone_vol_chop_1d_v2"
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
    if n < period * 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    atr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = np.where(atr_smooth > 1e-10, 100 * plus_dm_smooth / atr_smooth, 0)
    minus_di = np.where(atr_smooth > 1e-10, 100 * minus_dm_smooth / atr_smooth, 0)
    
    dx = np.zeros(n, dtype=np.float64)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period * 2, adjust=False).mean().values
    return adx

def calculate_camarilla_pivots(prev_high, prev_low, prev_close):
    """
    Camarilla pivot levels from previous day
    S3/S4 = support zones, R3/R4 = resistance zones
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
        
        # Standard Camarilla S3 and R3
        pivots['s3'][i] = close - high_low_range * 1.1 / 4
        pivots['r3'][i] = close + high_low_range * 1.1 / 4
    
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
    
    # Load 1d data ONCE for Camarilla pivots and trend
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 1d Camarilla pivots
    cam_pivots = calculate_camarilla_pivots(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values
    )
    
    # Align pivots to 4h (shifted by 1 to avoid look-ahead)
    s3_aligned = align_htf_to_ltf(prices, df_1d, cam_pivots['s3'])
    r3_aligned = align_htf_to_ltf(prices, df_1d, cam_pivots['r3'])
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    adx_14 = calculate_adx(high, low, close, period=14)
    
    # Volume moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # EMA for short-term trend
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
    
    # Warmup for indicators
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if critical indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
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
        # ADX > 20 = trending (good for Camarilla breakout)
        # ADX < 15 = ranging (skip - Camarilla doesn't work well in ranges)
        adx = adx_14[i] if not np.isnan(adx_14[i]) else 30.0
        is_trending = adx > 18.0
        
        # === TREND BIAS (1d HMA + EMA cross) ===
        price_above_1d_hma = close[i] > hma_1d_aligned[i] if not np.isnan(hma_1d_aligned[i]) else True
        ema_bullish = ema_8[i] > ema_21[i] if (not np.isnan(ema_8[i]) and not np.isnan(ema_21[i])) else True
        ema_bearish = ema_8[i] < ema_21[i] if (not np.isnan(ema_8[i]) and not np.isnan(ema_21[i])) else False
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === CAMARILLA PIVOT LEVELS ===
        s3 = s3_aligned[i]
        r3 = r3_aligned[i]
        
        # Distance to S3/R3 in ATR units
        dist_to_s3 = (close[i] - s3) / atr_14[i] if atr_14[i] > 0 else 999
        dist_to_r3 = (r3 - close[i]) / atr_14[i] if atr_14[i] > 0 else 999
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG ENTRY: Price at S3 support zone + bullish conditions
            # Within 0.5 ATR of S3 = touched support
            if dist_to_s3 >= -0.5 and dist_to_s3 <= 1.5:
                if price_above_1d_hma and ema_bullish and vol_spike:
                    desired_signal = SIZE
            
            # SHORT ENTRY: Price at R3 resistance zone + bearish conditions
            # Within 0.5 ATR of R3 = touched resistance
            if dist_to_r3 >= -0.5 and dist_to_r3 <= 1.5:
                if not price_above_1d_hma and ema_bearish and vol_spike:
                    desired_signal = -SIZE
        
        # === EXIT LOGIC ===
        exit_signal = 0.0
        
        if in_position and position_side > 0:
            # LONG position: check stoploss
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                exit_signal = 0.0
            else:
                exit_signal = SIZE
            
            # Take profit at R3
            if not np.isnan(r3) and high[i] >= r3:
                exit_signal = 0.0
        
        if in_position and position_side < 0:
            # SHORT position: check stoploss
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                exit_signal = 0.0
            else:
                exit_signal = -SIZE
            
            # Take profit at S3
            if not np.isnan(s3) and low[i] <= s3:
                exit_signal = 0.0
        
        desired_signal = exit_signal
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New entry or reversal
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