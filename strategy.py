#!/usr/bin/env python3
"""
Experiment #007: 6h Volatility Spike Mean Reversion

HYPOTHESIS: After extreme volatility spikes (ATR(7)/ATR(30) > 2.0), price often
reverts from BB extremes. Vol spike = panic/compression = snap-back opportunity.
Weekly trend filter avoids fighting major direction.

WHY THIS SHOULD WORK IN BOTH BULL AND BEAR:
- Bull markets: Vol spike + price at lower BB = panic dip = long entry
- Bear markets: Vol spike + price at upper BB = exhaustion rally = short entry
- Range markets: Vol spikes at BB extremes = reliable mean reversion
- Vol spike detection is market-agnostic (works in any condition)

KEY DESIGN:
1. ATR(7)/ATR(30) > 2.0 = volatility shock detected
2. Price at BB(20,2) lower/upper extreme = oversold/overbought
3. Weekly HMA for trend bias
4. Volume confirmation (>1.3x) to avoid false spikes
5. Signal: 0.25 discrete

TARGET: 75-150 total trades over 4 years (12-37/year)
DB reference: Vol spike reversion mentioned as working pattern
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_vol_spike_bb_reversion_1d_v1"
timeframe = "6h"
leverage = 1.0

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

def calculate_bollinger_bands(close, period=20, num_std=2):
    """Bollinger Bands - returns (lower, middle, upper)"""
    mid = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = mid + num_std * std
    lower = mid - num_std * std
    return lower, mid, upper

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
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
    
    atr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    for i in range(n):
        if atr_smooth[i] > 1e-10:
            plus_di[i] = 100 * plus_dm_smooth[i] / atr_smooth[i]
            minus_di[i] = 100 * minus_dm_smooth[i] / atr_smooth[i]
    
    dx = np.zeros(n)
    for i in range(n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 1d ADX for regime
    adx_1d_raw = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d_raw)
    
    # Calculate 6h indicators
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    
    # ATR ratio for volatility spike detection
    atr_ratio = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if not np.isnan(atr_30[i]) and atr_30[i] > 1e-10:
            atr_ratio[i] = atr_7[i] / atr_30[i]
    
    # Bollinger Bands
    bb_lower, bb_mid, bb_upper = calculate_bollinger_bands(close, period=20, num_std=2)
    
    # Volume moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # ADX for local trend
    adx_14 = calculate_adx(high, low, close, period=14)
    
    # ATR for position sizing
    atr_14 = calculate_atr(high, low, close, period=14)
    
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
    
    # Warmup for indicators
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_ratio[i]) or np.isnan(bb_lower[i]) or np.isnan(adx_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === VOL SPIKE DETECTION ===
        vol_spike = atr_ratio[i] > 2.0
        
        # === BB POSITION (0 = at lower band, 1 = at upper band) ===
        bb_range = bb_upper[i] - bb_lower[i]
        if bb_range > 1e-10:
            bb_position = (close[i] - bb_lower[i]) / bb_range
        else:
            bb_position = 0.5
        
        # === WEEKLY TREND FILTER ===
        weekly_bullish = True
        weekly_bearish = True
        if not np.isnan(hma_1d_aligned[i]) and hma_1d_aligned[i] > 1e-10:
            weekly_bullish = close[i] > hma_1d_aligned[i]
            weekly_bearish = close[i] < hma_1d_aligned[i]
        
        # === WEEKLY REGIME (use 1d ADX) ===
        weekly_trending = True
        if not np.isnan(adx_1d_aligned[i]):
            weekly_trending = adx_1d_aligned[i] > 20.0
        
        # === VOLUME CONFIRMATION ===
        vol_confirm = vol_ratio[i] > 1.3
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: Vol spike + price at/near lower BB + not weekly bearish
        # BB position < 0.15 = at/below lower band = potential reversal up
        if vol_spike and bb_position < 0.15 and not weekly_bearish:
            if vol_confirm:
                desired_signal = SIZE
            elif weekly_trending:
                desired_signal = SIZE
        
        # SHORT: Vol spike + price at/near upper BB + not weekly bullish
        # BB position > 0.85 = at/above upper band = potential reversal down
        if vol_spike and bb_position > 0.85 and not weekly_bullish:
            if vol_confirm:
                desired_signal = -SIZE
            elif weekly_trending:
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
        
        # === TAKE PROFIT ===
        tp_triggered = False
        
        if in_position and position_side > 0:
            # TP at middle BB or 2.5R profit
            profit_r = (close[i] - entry_price) / entry_atr
            if profit_r >= 2.5:
                tp_triggered = True
            if not np.isnan(bb_mid[i]) and close[i] >= bb_mid[i]:
                tp_triggered = True
        
        if in_position and position_side < 0:
            # TP at middle BB or 2.5R profit
            profit_r = (entry_price - close[i]) / entry_atr
            if profit_r >= 2.5:
                tp_triggered = True
            if not np.isnan(bb_mid[i]) and close[i] <= bb_mid[i]:
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