#!/usr/bin/env python3
"""
Experiment #012 v2: 12h Simple Breakout + Volume Confirmation + 1d Trend

HYPOTHESIS: 12h Donchian(20) channels mark institutional consolidation zones.
When price breaks outside the channel AND volume confirms (ratio > 1.5x 20-bar MA),
it signals a genuine move. The 1d HMA filters for trend direction, avoiding 
counter-trend entries. 12h is slow enough to avoid overtrading but fast enough 
to capture meaningful moves. The "price outside channel" condition (not strict 
cross) ensures enough trade opportunities while volume filter prevents whipsaws.

TIMEFRAME: 12h primary
HTF: 1d HMA for trend, 1w HMA for regime
TARGET: 75-150 total trades over 4 years (19-37/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_simple_breakout_vol_1d_v1"
timeframe = "12h"
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
    """Donchian Channel - returns upper and lower bands"""
    n = len(high)
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    mid = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
        mid[i] = (upper[i] + lower[i]) / 2.0
    
    return upper, lower, mid

def calculate_adx(high, low, close, period=14):
    """ADX for regime detection"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # True range
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Directional movement
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    # Smooth with EMA
    atr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # DI
    plus_di = 100 * plus_dm_smooth / (atr_smooth + 1e-10)
    minus_di = 100 * minus_dm_smooth / (atr_smooth + 1e-10)
    
    # DX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    
    # ADX
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # 1d HMA for trend direction
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # 1w HMA for regime (bull/bear/range)
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    donch_upper, donch_lower, donch_mid = calculate_donchian(high, low, period=20)
    
    # Volume ratio (current vs 20-bar MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # ADX for regime filter (not used in entry, but for confidence)
    adx_14 = calculate_adx(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_distance = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    profit_locked = False
    
    warmup = 100  # Need enough bars for Donchian(20) + HMA alignment
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === Trend Bias (1d HMA) ===
        price_above_1d_hma = close[i] > hma_1d_aligned[i]
        
        # === Regime (1w HMA) ===
        if np.isnan(hma_1w_aligned[i]):
            price_above_1w_hma = price_above_1d_hma  # Fallback to 1d
        else:
            price_above_1w_hma = close[i] > hma_1w_aligned[i]
        
        # === Volume Confirmation ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === ATR Volatility Filter ===
        # Ensure we're not in ultra-low volatility (false breakouts)
        atr_percent = atr_14[i] / close[i]
        min_volatility = atr_percent > 0.005  # ATR > 0.5% of price
        
        # === Donchian Channel Detection ===
        channel_width = donch_upper[i] - donch_lower[i]
        
        # Price position relative to channel
        price_in_channel = (close[i] > donch_lower[i]) and (close[i] < donch_upper[i])
        price_above_upper = close[i] > donch_upper[i]
        price_below_lower = close[i] < donch_lower[i]
        
        # Breakout strength: how far outside channel
        upper_breakout_pct = (close[i] - donch_upper[i]) / (donch_upper[i] + 1e-10) if price_above_upper else 0
        lower_breakout_pct = (donch_lower[i] - close[i]) / (donch_lower[i] + 1e-10) if price_below_lower else 0
        
        desired_signal = 0.0
        
        # === NEW ENTRY LOGIC ===
        if not in_position:
            # === LONG ENTRY ===
            # Price breaks above upper Donchian + volume spike + bullish trend
            if price_above_upper and vol_spike and min_volatility:
                if price_above_1d_hma:  # Trend aligned
                    desired_signal = SIZE
            
            # === SHORT ENTRY ===
            # Price breaks below lower Donchian + volume spike + bearish trend
            if price_below_lower and vol_spike and min_volatility:
                if not price_above_1d_hma:  # Trend aligned
                    desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR) ===
        if in_position:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = lowest_since_entry - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
            
            if position_side < 0:
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = highest_since_entry + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
        
        # === TRAILING STOP (3 ATR profit lock) ===
        if in_position and not profit_locked:
            if position_side > 0:
                profit_pct = (highest_since_entry - entry_price) / entry_price
                if profit_pct > 0.06:  # 6% profit = ~3 ATR for typical 12h
                    profit_locked = True
            if position_side < 0:
                profit_pct = (entry_price - lowest_since_entry) / entry_price
                if profit_pct > 0.06:
                    profit_locked = True
        
        # === PROFIT LOCKED: Tight trailing stop ===
        if in_position and profit_locked:
            if position_side > 0:
                # Trail stop: highest - 2 ATR
                trailing_stop = highest_since_entry - 2.0 * entry_atr
                if low[i] < trailing_stop:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                    profit_locked = False
            
            if position_side < 0:
                trailing_stop = lowest_since_entry + 2.0 * entry_atr
                if high[i] > trailing_stop:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                    profit_locked = False
        
        # === CHANNEL EXIT: Price re-enters channel ===
        if in_position:
            if position_side > 0 and price_in_channel:
                # Long: price fell back into channel = momentum lost
                desired_signal = 0.0
            if position_side < 0 and price_in_channel:
                # Short: price rallied back into channel = short covering
                desired_signal = 0.0
        
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
                profit_locked = False
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals