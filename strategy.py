#!/usr/bin/env python3
"""
Experiment #035: 6h Weekly Donchian Breakout + ADX + Volume

HYPOTHESIS: Weekly Donchian breakout captures major institutional trend changes.
Weekly ADX (not choppiness) confirms momentum. Weekly volume confirms smart money.
Weekly RSI avoids entries at extended levels.

WHY THIS SHOULD WORK IN BOTH BULL AND BEAR:
- Weekly Donchian is pure price structure — symmetric for bull/bear
- Bull: long breakouts above weekly high, ride with trailing ATR stop
- Bear: short breakouts below weekly low ( rallies to resistance)
- 6h captures enough for 50-150 trades/year without overtrading

KEY DIFFERENCES FROM FAILED ATTEMPTS:
- NOT Camarilla (failed 4+ times with overtrading/neg Sharpe)
- NOT Choppiness (caused 0 trades in multiple attempts)
- Weekly-only structure for signal (not mixed HTF/LTF)
- ADX instead of choppiness for regime (more reliable)

TARGET: 75-120 total trades over 4 years = 19-30/year.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_weekly_donchian_adx_vol_v1"
timeframe = "6h"
leverage = 1.0

def calculate_adx(high, low, close, period=14):
    """Calculate ADX using proper Wilder smoothing"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)
    
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / np.where(atr > 0, atr, 1) * 100
    minus_di = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / np.where(atr > 0, atr, 1) * 100
    
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10) * 100
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    return pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_rsi(close, period=14):
    """RSI indicator"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_ema(close, span):
    """Exponential Moving Average"""
    return pd.Series(close).ewm(span=span, min_periods=span, adjust=False).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load weekly data ONCE
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly indicators (calculated on weekly data)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_volume = df_1w['volume'].values
    
    # Weekly Donchian (20 bars = ~5 trading weeks)
    donchian_period = 20
    weekly_donch_upper = np.full(len(weekly_close), np.nan, dtype=np.float64)
    weekly_donch_lower = np.full(len(weekly_close), np.nan, dtype=np.float64)
    for i in range(donchian_period - 1, len(weekly_close)):
        weekly_donch_upper[i] = np.max(weekly_high[i - donchian_period + 1:i + 1])
        weekly_donch_lower[i] = np.min(weekly_low[i - donchian_period + 1:i + 1])
    
    # Weekly ADX
    weekly_adx = calculate_adx(weekly_high, weekly_low, weekly_close, period=14)
    
    # Weekly RSI
    weekly_rsi = calculate_rsi(weekly_close, period=14)
    
    # Weekly EMA for trend
    weekly_ema_fast = calculate_ema(weekly_close, 8)
    weekly_ema_slow = calculate_ema(weekly_close, 21)
    
    # Weekly volume MA
    vol_ma_period = 20
    weekly_vol_ma = np.full(len(weekly_volume), np.nan, dtype=np.float64)
    for i in range(vol_ma_period - 1, len(weekly_volume)):
        weekly_vol_ma[i] = np.mean(weekly_volume[i - vol_ma_period + 1:i + 1])
    weekly_vol_ratio = weekly_volume / np.where(weekly_vol_ma > 0, weekly_vol_ma, 1)
    
    # Align all weekly indicators to 6h
    donch_upper_aligned = align_htf_to_ltf(prices, df_1w, weekly_donch_upper)
    donch_lower_aligned = align_htf_to_ltf(prices, df_1w, weekly_donch_lower)
    adx_aligned = align_htf_to_ltf(prices, df_1w, weekly_adx)
    rsi_aligned = align_htf_to_ltf(prices, df_1w, weekly_rsi)
    ema_fast_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema_fast)
    ema_slow_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema_slow)
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1w, weekly_vol_ratio)
    
    # 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    ema_8 = calculate_ema(close, 8)
    ema_21 = calculate_ema(close, 21)
    
    # 6h volume for confirmation
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio_6h = volume / np.where(vol_ma_6h > 0, vol_ma_6h, 1)
    
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
    
    warmup = 80
    
    for i in range(warmup, n):
        # Check indicators ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 0:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # Skip if weekly data not ready
        donch_upper = donch_upper_aligned[i]
        donch_lower = donch_lower_aligned[i]
        adx_val = adx_aligned[i]
        rsi_val = rsi_aligned[i]
        vol_ratio_w = vol_ratio_aligned[i]
        vol_ratio_6 = vol_ratio_6h[i]
        ema_fast = ema_fast_aligned[i]
        ema_slow = ema_slow_aligned[i]
        
        if np.isnan(donch_upper) or np.isnan(donch_lower):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(adx_val) or np.isnan(rsi_val):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(vol_ratio_w) or np.isnan(vol_ratio_6):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # Weekly trend (EMA alignment)
        weekly_bullish = ema_fast > ema_slow if not (np.isnan(ema_fast) or np.isnan(ema_slow)) else True
        weekly_bearish = ema_fast < ema_slow if not (np.isnan(ema_fast) or np.isnan(ema_slow)) else False
        
        # 6h EMA for entry timing
        ema_bullish_6h = ema_8[i] > ema_21[i] if not (np.isnan(ema_8[i]) or np.isnan(ema_21[i])) else True
        ema_bearish_6h = ema_8[i] < ema_21[i] if not (np.isnan(ema_8[i]) or np.isnan(ema_21[i])) else False
        
        # ADX trend confirmation (must be trending, not ranging)
        is_trending = adx_val > 22.0
        
        # RSI not extended (avoid buying tops, shorting bottoms)
        rsi_not_high = rsi_val < 72.0
        rsi_not_low = rsi_val > 28.0
        
        # Volume confirmation (spike on weekly OR 6h)
        vol_spike_w = vol_ratio_w > 1.4
        vol_spike_6 = vol_ratio_6 > 1.3
        
        # Weekly Donchian breakout signals
        price_above_donch = close[i] > donch_upper
        price_below_donch = close[i] < donch_lower
        
        # Distance to breakout level (in ATR units)
        dist_to_upper = (donch_upper - close[i]) / atr_14[i] if atr_14[i] > 0 else 999
        dist_to_lower = (close[i] - donch_lower) / atr_14[i] if atr_14[i] > 0 else 999
        
        desired_signal = 0.0
        
        # === ENTRY LOGIC ===
        # LONG: Weekly breakout above, bullish EMA, ADX trending, RSI not extended
        if not in_position or position_side <= 0:
            if price_above_donch:
                if weekly_bullish and is_trending and rsi_not_high:
                    if vol_spike_w or vol_spike_6:
                        desired_signal = SIZE
                    elif ema_bullish_6h:
                        desired_signal = SIZE
        
        # SHORT: Weekly breakdown below, bearish EMA, ADX trending, RSI not extended
        if not in_position or position_side >= 0:
            if price_below_donch:
                if weekly_bearish and is_trending and rsi_not_low:
                    if vol_spike_w or vol_spike_6:
                        desired_signal = -SIZE
                    elif ema_bearish_6h:
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
        
        # === REVERSAL: Exit and flip ===
        # If we have a signal in opposite direction while in position
        if in_position:
            if position_side > 0 and desired_signal < 0:
                # Stopped out or reversal signal
                desired_signal = 0.0  # Force flat first
            elif position_side < 0 and desired_signal > 0:
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