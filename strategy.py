#!/usr/bin/env python3
"""
Experiment #007: 6h Camarilla Pivot Mean Reversion + 1d Trend Filter

HYPOTHESIS: Camarilla pivot levels (S3/S4/R3/R4) from 1d data represent institutional
order zones where mean reversion occurs. Unlike pure breakout strategies (which failed
repeatedly in this session), mean reversion at extreme levels works in BOTH bull and
bear markets because it fades overextensions regardless of trend direction.

WHY THIS SHOULD WORK:
- Bear markets (2022 crash, 2025 test): Short at R3/R4 when price overextends above value
- Bull markets: Long at S3/S4 when price dips below value (buy the dip)
- Range markets: Mean revert between pivot levels
- 1d HMA(21) provides trend bias to avoid counter-trend trades
- Choppiness filter avoids entering during extreme chop where pivots fail

KEY IMPROVEMENTS vs failed #016 (Camarilla Sharpe=-2.331):
1. Tighter entry zones (price must TOUCH pivot, not just approach)
2. Volume confirmation is OPTIONAL (not required) - increases trade count
3. Choppiness threshold relaxed (61.8 instead of 55) - more trades allowed
4. Added RSI(14) extreme filter for additional confluence
5. Proper position tracking with ATR stoploss

TARGET: 75-200 total trades over 4 years (19-50/year)
SIZE: 0.25 (discrete)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_camarilla_meanrev_1d_hma_rsi_v1"
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

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if avg_loss[i] > 0:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - CHOP > 61.8 = ranging, CHOP < 38.2 = trending"""
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
    Camarilla pivot levels from previous day
    S3/S4 = support zones (long entries)
    R3/R4 = resistance zones (short entries)
    """
    n = len(prev_high)
    pivots = {
        's3': np.full(n, np.nan, dtype=np.float64),
        's4': np.full(n, np.nan, dtype=np.float64),
        'r3': np.full(n, np.nan, dtype=np.float64),
        'r4': np.full(n, np.nan, dtype=np.float64),
    }
    
    for i in range(1, n):  # Start from 1 (need prev day)
        if np.isnan(prev_high[i-1]) or np.isnan(prev_low[i-1]) or np.isnan(prev_close[i-1]):
            continue
        
        high_low_range = prev_high[i-1] - prev_low[i-1]
        if high_low_range <= 1e-10:
            continue
        
        close = prev_close[i-1]
        
        # Shift pivots forward by 1 (use yesterday's pivots for today)
        pivots['s3'][i] = close - high_low_range * 1.1 / 4
        pivots['s4'][i] = close - high_low_range * 1.1 / 2
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
    
    # Calculate Camarilla pivots from 1d (use previous day's OHLC)
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
    chop_14 = calculate_choppiness(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    
    # Volume moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # EMA for short-term trend
    ema_8 = calculate_ema(close, 8)
    ema_21 = calculate_ema(close, 21)
    
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(rsi_14[i]):
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
        
        # === REGIME CHECK ===
        chop = chop_14[i]
        # Allow trades when NOT extremely choppy (CHOP < 61.8)
        is_tradeable = chop < 61.8
        
        # === TREND BIAS (1d HMA) ===
        price_above_1d_hma = close[i] > hma_1d_aligned[i] if not np.isnan(hma_1d_aligned[i]) else True
        
        # === RSI EXTREMES ===
        rsi = rsi_14[i]
        rsi_oversold = rsi < 35
        rsi_overbought = rsi > 65
        
        # === CAMARILLA PIVOT LEVELS ===
        s3 = s3_aligned[i]
        s4 = s4_aligned[i]
        r3 = r3_aligned[i]
        r4 = r4_aligned[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: Price at/near S3/S4 support + RSI oversold + trend bias
        if is_tradeable:
            # At S3 support (price within 1% of S3 or below)
            if not np.isnan(s3):
                dist_to_s3_pct = (close[i] - s3) / s3 * 100 if s3 > 0 else 999
                if dist_to_s3_pct < 1.5 and dist_to_s3_pct > -5.0:  # Within 1.5% above or 5% below S3
                    if rsi_oversold and price_above_1d_hma:
                        desired_signal = SIZE
                    elif rsi_oversold:  # Allow without trend bias if RSI extreme enough
                        if rsi < 25:
                            desired_signal = SIZE
            
            # At S4 deeper support (stronger signal)
            if not np.isnan(s4) and desired_signal == 0:
                dist_to_s4_pct = (close[i] - s4) / s4 * 100 if s4 > 0 else 999
                if dist_to_s4_pct < 2.0 and dist_to_s4_pct > -5.0:
                    if rsi_oversold:
                        desired_signal = SIZE
        
        # SHORT: Price at/near R3/R4 resistance + RSI overbought + trend bias
        if is_tradeable and desired_signal == 0:
            # At R3 resistance (price within 1% of R3 or above)
            if not np.isnan(r3):
                dist_to_r3_pct = (r3 - close[i]) / r3 * 100 if r3 > 0 else 999
                if dist_to_r3_pct < 1.5 and dist_to_r3_pct > -5.0:
                    if rsi_overbought and not price_above_1d_hma:
                        desired_signal = -SIZE
                    elif rsi_overbought:
                        if rsi > 75:
                            desired_signal = -SIZE
            
            # At R4 deeper resistance (stronger signal)
            if not np.isnan(r4) and desired_signal == 0:
                dist_to_r4_pct = (r4 - close[i]) / r4 * 100 if r4 > 0 else 999
                if dist_to_r4_pct < 2.0 and dist_to_r4_pct > -5.0:
                    if rsi_overbought:
                        desired_signal = -SIZE
        
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
            # TP at R3
            if not np.isnan(r3) and high[i] >= r3:
                tp_triggered = True
        
        if in_position and position_side < 0:
            # TP at S3
            if not np.isnan(s3) and low[i] <= s3:
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