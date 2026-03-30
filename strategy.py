#!/usr/bin/env python3
"""
Experiment #007: 6h Donchian Breakout + Weekly EMA Trend + Volume

HYPOTHESIS: Donchian channel breakouts capture explosive moves when price 
congestion resolves. By using 1w EMA for trend direction (bull market = only 
long breakouts, bear market = only short breakouts), we avoid getting caught 
in choppy whipsaws during transitions.

WHY 6h: Balances signal quality with trade frequency. 6h is 50% faster than 
12h, giving more opportunities while still filtering noise compared to 4h.

WHY IT WORKS IN BULL AND BEAR: Weekly EMA is slow enough to identify the 
dominant multi-month trend. Long breakouts in bull markets, short breakouts 
in bear markets. The weekly filter prevents buying breakouts at bear market 
tops and shorting breakouts at bull market bottoms.

TARGET: 75-200 total trades over 4 years = 19-50/year.
Signal size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_donchian_weekly_ema_vol_v1"
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA(21) for multi-month trend direction
    ema_1w = pd.Series(df_1w['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Weekly ADX for trend strength confirmation
    htf_high = df_1w['high'].values
    htf_low = df_1w['low'].values
    htf_close = df_1w['close'].values
    adx_1w = calculate_adx(htf_high, htf_low, htf_close, period=14)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # === Local 6h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian channels (lookback 20 = 5 days at 6h)
    donchian_period = 20
    highest_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Local ADX for entry timing (shorter period = faster signal)
    adx_6h = calculate_adx(high, low, close, period=10)
    
    # Signals
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
    entry_bar = 0
    
    warmup = max(150, donchian_period + 21)  # Need enough for Donchian + weekly EMA
    
    for i in range(warmup, n):
        # Skip if ATR not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if weekly indicators not aligned
        if np.isnan(ema_1w_aligned[i]) or np.isnan(adx_1w_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1w EMA21) ===
        weekly_trend_up = close[i] > ema_1w_aligned[i]
        weekly_trend_strength = adx_1w_aligned[i] > 20  # Weak = choppy
        
        # === LOCAL TREND (6h ADX) ===
        local_adx = adx_6h[i] if not np.isnan(adx_6h[i]) else 0
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.3
        
        # Donchian breakout levels from previous bar
        prev_donch_high = highest_high[i - 1] if i > 0 else highest_high[i]
        prev_donch_low = lowest_low[i - 1] if i > 0 else lowest_low[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Price breaks above 20-period high with weekly trend alignment ===
            if weekly_trend_up and weekly_trend_strength:
                # Breakout above Donchian high
                if high[i] > prev_donch_high:
                    desired_signal = SIZE
            
            # === SHORT: Price breaks below 20-period low with weekly trend alignment ===
            if not weekly_trend_up and weekly_trend_strength:
                # Breakdown below Donchian low
                if low[i] < prev_donch_low:
                    desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR trailing) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                desired_signal = 0.0
        
        # === HOLD PERIOD (minimum 3 bars = 18h to avoid churn) ===
        bars_held = i - entry_bar
        
        if in_position and bars_held >= 3:
            # Partial take profit at 2:1 reward (reduce to half position)
            if position_side > 0:
                profit_pct = (close[i] - entry_price) / entry_price
                if profit_pct >= 0.05:  # 5% profit (about 2x ATR typically)
                    desired_signal = SIZE / 2  # Take half off the table
            if position_side < 0:
                profit_pct = (entry_price - close[i]) / entry_price
                if profit_pct >= 0.05:
                    desired_signal = -SIZE / 2
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                entry_bar = i
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                stop_price = 0.0
        
        signals[i] = desired_signal
    
    return signals


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
    up_move = np.zeros(n, dtype=np.float64)
    down_move = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        up_move[i] = high[i] - high[i-1] if (high[i] - high[i-1]) > (low[i-1] - low[i]) else 0
        down_move[i] = low[i-1] - low[i] if (low[i-1] - low[i]) > (high[i] - high[i-1]) else 0
    
    # Smoothed TR
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Smoothed +DM
    up_smooth = pd.Series(up_move).ewm(span=period, min_periods=period, adjust=False).mean().values
    down_smooth = pd.Series(down_move).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # DI
    di_plus = np.where(tr_smooth > 0, 100 * up_smooth / tr_smooth, 0)
    di_minus = np.where(tr_smooth > 0, 100 * down_smooth / tr_smooth, 0)
    
    # DX
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    
    # ADX
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx