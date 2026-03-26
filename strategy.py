#!/usr/bin/env python3
"""
Experiment #003 Rev2: 4h Donchian Breakout + Volume + 1d HMA Trend

HYPOTHESIS: Donchian channel breakouts capture momentum moves when price breaks
recent highs/lows. Volume confirmation ensures institutional participation.
1d HMA provides trend bias to avoid counter-trend trades that fail in bear markets.

WHY THIS SHOULD WORK IN BOTH BULL AND BEAR:
- Bull markets: Long breakouts above Donchian(20) high when 1d HMA sloping up
- Bear markets: Short breakouts below Donchian(20) low when 1d HMA sloping down
- Volume spike (>1.3x avg) confirms breakout validity, reduces false signals
- ATR trailing stop protects capital during reversals

TARGET: 100-200 total trades over 4 years (25-50/year)
DB reference: mtf_4h_hma_donchian_volume_rsi_12h_atr_v1 (Sharpe=1.382, 95tr)

KEY DESIGN:
1. Donchian(20) breakout as primary trigger (proven pattern)
2. Volume > 1.3x 20-avg (lower threshold = more trades than 1.5x)
3. 1d HMA slope for trend bias (only trade with HTF trend)
4. ATR(14) trailing stop at 2.0x
5. Signal: 0.30 (discrete, 30% position size)

SIMPLIFIED FROM PREVIOUS: Removed choppiness filter (was over-filtering),
lowered volume threshold from 1.5x to 1.3x, removed Camarilla pivots (too rare).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_vol_hma_1d_v2"
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period"""
    n = len(high)
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_ema(close, span):
    """Exponential Moving Average"""
    return pd.Series(close).ewm(span=span, min_periods=span, adjust=False).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 1d HMA slope (current vs 3 bars ago on 1d)
    hma_1d_slope = np.full(n, np.nan, dtype=np.float64)
    for i in range(3, n):
        if not np.isnan(hma_1d_aligned[i]) and not np.isnan(hma_1d_aligned[i-3]):
            hma_1d_slope[i] = hma_1d_aligned[i] - hma_1d_aligned[i-3]
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Volume moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # EMA for additional confirmation
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
    
    # Warmup
    warmup = 60
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
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
        
        # === TREND BIAS (1d HMA slope) ===
        hma_slope = hma_1d_slope[i]
        trend_bullish = hma_slope > 0 if not np.isnan(hma_slope) else True
        trend_bearish = hma_slope < 0 if not np.isnan(hma_slope) else False
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.3  # Lower threshold for more trades
        
        # === DONCHIAN BREAKOUT DETECTION ===
        prev_upper = donchian_upper[i-1] if i > 0 else donchian_upper[i]
        prev_lower = donchian_lower[i-1] if i > 0 else donchian_lower[i]
        
        # Breakout above upper channel
        breakout_long = close[i] > prev_upper and high[i] > prev_upper
        
        # Breakout below lower channel
        breakout_short = close[i] < prev_lower and low[i] < prev_lower
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: Donchian breakout + volume + bullish trend
        if breakout_long and vol_spike and trend_bullish:
            desired_signal = SIZE
        
        # SHORT: Donchian breakout + volume + bearish trend
        if breakout_short and vol_spike and trend_bearish:
            desired_signal = -SIZE
        
        # === STOPLOSS CHECK (ATR trailing) ===
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
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
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