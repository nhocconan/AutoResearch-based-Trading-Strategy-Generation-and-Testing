#!/usr/bin/env python3
"""
Experiment #004: 1d Donchian Breakout + 1w Trend Filter

HYPOTHESIS: Simple Donchian(20) breakout on 1d with 1w trend confirmation.
1w HMA filters out countertrend trades, volume spike confirms institutional moves.
This is the proven pattern from DB: mtf_1d_kama_rsi_chop_regime_1w_v1 (Sharpe=1.31, 74 trades).

WHY THIS SHOULD WORK:
- 1d timeframe naturally limits trades to 30-80 over 4 years
- Donchian breakout is robust across bull/bear/range markets
- 1w trend filter prevents fighting major trends
- Tight entry (breakout + volume + 1w trend) = high win rate

TARGET: 50-80 total trades over 4 years (proven sustainable)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_breakout_1w_v1"
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - returns upper, lower, mid"""
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2.0
    return upper, lower, mid

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness
    CHOP > 61.8 = ranging (avoid), CHOP < 50 = trending (allow)
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA for trend
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 1w EMA for additional confirmation
    ema_1w_raw = pd.Series(df_1w['close'].values).ewm(span=8, min_periods=8, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_raw)
    
    # Calculate 1d indicators
    donch_upper, donch_lower, donch_mid = calculate_donchian(high, low, period=20)
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
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup - need 20 bars for Donchian, 21 for HMA
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
        
        if np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1w_aligned[i]) or np.isnan(ema_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === 1w TREND CHECK ===
        price_above_1w_hma = close[i] > hma_1w_aligned[i]
        price_above_1w_ema = close[i] > ema_1w_aligned[i]
        ema_above_hma = ema_1w_aligned[i] > hma_1w_aligned[i]
        
        # 1w bullish: price above both HMA and EMA, EMA above HMA
        weekly_bullish = price_above_1w_hma and price_above_1w_ema and ema_above_hma
        # 1w bearish: price below both HMA and EMA, EMA below HMA
        weekly_bearish = not price_above_1w_hma and not price_above_1w_ema and not ema_above_hma
        
        # === REGIME CHECK ===
        chop = chop_14[i]
        is_trending = chop < 55.0
        
        # === DONCHIAN BREAKOUT ===
        upper = donch_upper[i]
        lower = donch_lower[i]
        
        # Check if price broke above upper band (bullish breakout)
        bullish_breakout = high[i] >= upper and close[i] > open[i] if 'open' in prices.columns else high[i] >= upper
        # Check if price broke below lower band (bearish breakout)
        bearish_breakout = low[i] <= lower and close[i] < open[i] if 'open' in prices.columns else low[i] <= lower
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: Bullish breakout + weekly trend confirms + volume + trending regime
        if bullish_breakout and weekly_bullish and is_trending:
            if vol_spike:
                desired_signal = SIZE
            else:
                desired_signal = SIZE * 0.5  # Half size without volume confirmation
        
        # SHORT: Bearish breakout + weekly trend confirms + volume + trending regime
        if bearish_breakout and weekly_bearish and is_trending:
            if vol_spike:
                desired_signal = -SIZE
            else:
                desired_signal = -SIZE * 0.5  # Half size without volume confirmation
        
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
            # Reset position
            in_position = False
            position_side = 0
            entry_atr = 0.0
            stop_price = 0.0
            highest_since_entry = 0.0
            lowest_since_entry = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = low[i] - 2.0 * entry_atr
                else:
                    stop_price = high[i] + 2.0 * entry_atr
        
        signals[i] = desired_signal
    
    return signals