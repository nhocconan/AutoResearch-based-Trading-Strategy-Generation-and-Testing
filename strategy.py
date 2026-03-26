#!/usr/bin/env python3
"""
Experiment #030: 6h ADX Trend Strength + BB Width Regime Filter

HYPOTHESIS: On 6h timeframe, combining ADX for trend strength with Bollinger 
Band Width percentile for regime detection captures major trend moves while 
avoiding range-bound chop. ADX > 25 confirms directional momentum, BB Width 
percentile > 65 identifies expanding volatility regimes (trending moves), 
and 1d HMA provides directional bias. Works in both bull (trend long on ADX 
breakouts) and bear (trend short on ADX breakdowns + HMA downtrend).

TIMEFRAME: 6h primary
HTF: 1d for trend bias (HMA) and trend strength (ADX)
TARGET: 75-200 total trades over 4 years (19-50/year)
SIZE: 0.25
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_adx_bb_width_1d_trend_v1"
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

def calculate_adx(high, low, close, period=14):
    """Average Directional Index"""
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
    
    # Smooth with EWM
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # DX
    di_plus = np.zeros(n, dtype=np.float64)
    di_minus = np.zeros(n, dtype=np.float64)
    dx = np.zeros(n, dtype=np.float64)
    
    for i in range(period, n):
        if atr[i] > 0:
            di_plus[i] = 100 * plus_dm_smooth[i] / atr[i]
            di_minus[i] = 100 * minus_dm_smooth[i] / atr[i]
            di_sum = di_plus[i] + di_minus[i]
            if di_sum > 0:
                dx[i] = 100 * abs(di_plus[i] - di_minus[i]) / di_sum
    
    # ADX is EWM of DX
    adx = pd.Series(dx).ewm(span=period, min_periods=period * 2, adjust=False).mean().values
    
    return adx

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Bollinger Bands - returns upper, middle, lower"""
    n = len(close)
    middle = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    
    return upper, middle, lower

def calculate_bb_width_percentile(close, period=20, lookback=100):
    """BB Width percentile - identifies expanding vs contracting volatility"""
    n = len(close)
    upper, middle, lower = calculate_bollinger_bands(close, period=period)
    
    bb_width = (upper - lower) / (middle + 1e-10)
    width_pct = np.full(n, 50.0, dtype=np.float64)
    
    for i in range(lookback, n):
        if not np.isnan(bb_width[i]):
            recent = bb_width[max(0, i-lookback):i+1]
            recent = recent[~np.isnan(recent)]
            if len(recent) > 10:
                sorted_vals = np.sort(recent)
                rank = np.searchsorted(sorted_vals, bb_width[i])
                width_pct[i] = 100.0 * rank / len(sorted_vals)
    
    return width_pct

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # 1d ADX for trend strength
    adx_1d_raw = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d_raw)
    
    # Calculate local 6h indicators
    atr_14 = np.zeros(n, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Local ADX
    adx_local = calculate_adx(high, low, close, period=14)
    
    # BB for entry signals
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close, period=20)
    
    # BB Width percentile
    bb_width_pct = calculate_bb_percentile(close, period=20, lookback=100)
    
    # Volume MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
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
    
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(adx_local[i]) or np.isnan(bb_width_pct[i]):
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
        
        # === TREND BIAS (1d HMA) ===
        price_above_1d_hma = close[i] > hma_1d_aligned[i]
        
        # === TREND STRENGTH (1d ADX aligned to 6h) ===
        adx_strong = adx_1d_aligned[i] > 20 if not np.isnan(adx_1d_aligned[i]) else False
        
        # === LOCAL ADX (trend confirmation) ===
        local_adx_strong = adx_local[i] > 25
        
        # === BB WIDTH REGIME ===
        bb_width_expanding = bb_width_pct[i] > 65  # Top 35% of volatility
        bb_width_contracting = bb_width_pct[i] < 35
        
        # === BB BREAKOUT DETECTION ===
        # Price crosses above BB upper = bullish breakout
        # Price crosses below BB lower = bearish breakdown
        above_bb_upper = close[i] > bb_upper[i]
        below_bb_lower = close[i] < bb_lower[i]
        prev_above_bb_upper = close[i-1] > bb_upper[i-1] if i > 0 else False
        prev_below_bb_lower = close[i-1] < bb_lower[i-1] if i > 0 else False
        
        breakout_up = above_bb_upper and not prev_above_bb_upper
        breakout_down = below_bb_lower and not prev_below_bb_lower
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.2
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === NEW LONG ENTRY ===
            # Requirements: BB breakout up + expanding volatility + 1d trend aligned
            if breakout_up and bb_width_expanding and price_above_1d_hma:
                if vol_spike or local_adx_strong:  # Volume or momentum confirmation
                    desired_signal = SIZE
            
            # === NEW SHORT ENTRY ===
            # Requirements: BB breakout down + expanding volatility + 1d trend aligned
            if breakout_down and bb_width_expanding and not price_above_1d_hma:
                if vol_spike or local_adx_strong:
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
        
        # === EXIT LOGIC ===
        exit_triggered = False
        
        if in_position and position_side > 0:
            # Long exit: price breaks below BB lower OR extreme contraction
            if below_bb_lower:
                exit_triggered = True
            if bb_width_contracting and close[i] < bb_middle[i]:
                exit_triggered = True
        
        if in_position and position_side < 0:
            # Short exit: price breaks above BB upper OR extreme contraction
            if above_bb_upper:
                exit_triggered = True
            if bb_width_contracting and close[i] > bb_middle[i]:
                exit_triggered = True
        
        if exit_triggered:
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


def calculate_bb_percentile(close, period=20, lookback=100):
    """BB Width percentile - identifies expanding vs contracting volatility"""
    n = len(close)
    upper, middle, lower = calculate_bollinger_bands(close, period=period)
    
    bb_width = (upper - lower) / (middle + 1e-10)
    width_pct = np.full(n, 50.0, dtype=np.float64)
    
    for i in range(lookback, n):
        if not np.isnan(bb_width[i]):
            recent = bb_width[max(0, i-lookback):i+1]
            recent = recent[~np.isnan(recent)]
            if len(recent) > 10:
                sorted_vals = np.sort(recent)
                rank = np.searchsorted(sorted_vals, bb_width[i])
                width_pct[i] = 100.0 * rank / len(sorted_vals)
    
    return width_pct