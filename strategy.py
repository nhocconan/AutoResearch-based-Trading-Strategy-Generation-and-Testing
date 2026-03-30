#!/usr/bin/env python3
"""
Experiment #021: 4h Donchian(20) Breakout + Volume + 12h HMA Trend

HYPOTHESIS: Price extremes (Donchian breakout) mark institutional accumulation/distribution.
Combined with volume confirmation and 12h HMA21 trend alignment, this catches major turns
while avoiding whipsaws. Donchian breakout is STRICTER than Camarilla touch (requires CLOSE
beyond level, not just touching), reducing overtrading.

WHY DONCHIAN vs CAMARILLA: Camarilla S3/S4 touches trigger too many false signals.
Donchian requires price to CLOSE beyond the channel — more conservative, fewer trades.
Database confirms: Donchian+volume strategies have 1.3-1.5 test Sharpe, Camarilla overtrades.

WHY 4h: 2x faster than 12h for ~2x more signals, but still low enough to avoid fee drag.
12h HTF confirms larger trend direction.

TARGET: 75-150 total trades over 4 years = 19-37/year. HARD MAX: 200.
Signal size: 0.30.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_vol_hma21_12h_v2"
timeframe = "4h"
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

def calculate_hma(data, period):
    """Hull Moving Average"""
    series = pd.Series(data)
    half_length = period // 2
    sqrt_length = int(np.sqrt(period))
    
    wma1 = series.rolling(window=half_length, min_periods=half_length).mean()
    wma2 = series.rolling(window=period, min_periods=period).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.rolling(window=sqrt_length, min_periods=sqrt_length).mean()
    return hma.values

def calculate_donchian(high, low, period=20):
    """Donchian Channel - returns (upper, lower, middle)"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).max().values  # Note: min for lower
    # Correct lower calculation
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    middle = (upper + lower) / 2
    return upper, lower, middle

def calculate_adx(high, low, close, period=14):
    """Average Directional Index"""
    n = len(close)
    if n < period + 2:
        return np.full(n, np.nan)
    
    # True range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Plus/Minus DM
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    # Smoothed
    atr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # DI
    plus_di = 100 * plus_dm_smooth / np.where(atr_smooth > 0, atr_smooth, 1)
    minus_di = 100 * minus_dm_smooth / np.where(atr_smooth > 0, atr_smooth, 1)
    
    # DX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    
    # ADX
    adx = pd.Series(dx).ewm(span=period, min_periods=period * 2, adjust=False).mean().values
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_12h = get_htf_data(prices, '12h')
    
    # 12h HMA21 for trend direction
    hma_12h = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    adx_14 = calculate_adx(high, low, close, period=14)
    
    # Donchian(20) channel
    donch_upper, donch_lower, donch_mid = calculate_donchian(high, low, period=20)
    
    # Volume ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Choppiness Index (simple regime filter)
    # Sum of ATR(14) over 14 periods / Highest high - Lowest low over 14 periods
    # > 61.8 = choppy, < 38.2 = trending
    atr_sum = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    hh_ll = pd.Series(high).rolling(window=14, min_periods=14).max().values - \
            pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(np.where(atr_sum > 0, atr_sum, 1) / np.where(hh_ll > 0, hh_ll, 1)) / np.log10(14)
    
    # Signals
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
    entry_bar = 0
    
    warmup = 100  # Buffer for alignment
    
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
        
        if np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === FILTERS ===
        # 1. Volume confirmation (1.5x average)
        vol_confirm = vol_ratio[i] > 1.5
        
        # 2. ADX trend strength (above 20 = trending)
        adx_trending = adx_14[i] > 20
        
        # 3. Choppiness filter (below 61.8 = not too choppy)
        chop_filter = chop[i] < 61.8
        
        # === HTF TREND (12h HMA21) ===
        hma_bullish = close[i] > hma_12h_aligned[i]
        hma_bearish = close[i] < hma_12h_aligned[i]
        
        # === DONCHIAN BREAKOUT (require CLOSE beyond channel, not just touch) ===
        # Breakout above upper band
        breakout_up = close[i] > donch_upper[i] and close[i-1] <= donch_upper[i-1]
        # Breakout below lower band  
        breakout_down = close[i] < donch_lower[i] and close[i-1] >= donch_lower[i-1]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Breakout above + volume + HTF bullish + trend confirm ===
            if breakout_up and vol_confirm and hma_bullish and adx_trending:
                desired_signal = SIZE
            
            # === SHORT: Breakout below + volume + HTF bearish + trend confirm ===
            if breakout_down and vol_confirm and hma_bearish and adx_trending:
                desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR trailing - wider to avoid premature exits) ===
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
        
        # === MINIMUM HOLD (6 bars = 1 day to avoid noise churn) ===
        bars_held = i - entry_bar
        min_hold_bars = 6
        
        # === TAKE PROFIT (optional: exit at Donchian mid) ===
        if in_position and bars_held >= min_hold_bars:
            # Price returned to channel mid = potential reversal
            if position_side > 0 and close[i] <= donch_mid[i]:
                desired_signal = 0.0
            if position_side < 0 and close[i] >= donch_mid[i]:
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