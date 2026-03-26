#!/usr/bin/env python3
"""
Experiment #003: 4h Donchian Breakout + Volume Spike + Choppiness + 12h Trend Bias

HYPOTHESIS: Price channels (Donchian) capture institutional breakout moments.
Volume spike confirms the move is institutional, not noise. Choppiness filters
out ranging periods where breakouts fail. 12h KAMA provides trend bias to avoid
counter-trend entries during major market direction changes.

WHY THIS SHOULD WORK IN BOTH BULL AND BEAR:
- Donchian breakouts work in all markets (bull breaks up, bear breaks down)
- Bear markets: shorts trigger on breakdowns with tight ATR stop
- Bull markets: longs trigger on breakouts with trailing stop
- Range markets: choppiness filter prevents false breakouts

TARGET: 75-150 total trades over 4 years (19-38/year).
DB references:
  - mtf_4h_hma_donchian_volume_rsi_12h_atr_v1 (Sharpe=1.382, 95 tr)
  - mtf_4h_hma_donchian_volume_rsi_12h_atr_v1 (Sharpe=1.322, 94 tr)

KEY DESIGN:
1. Donchian(20) breakout as primary signal
2. Volume spike confirmation (>1.5x 20-avg)
3. Choppiness filter (CHOP < 61.8 = trending mode)
4. 12h KAMA for trend direction bias
5. ATR(14) stoploss (2x ATR)
6. Signal: 0.30 (discrete), max 0.40
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_vol_chop_12h_v1"
timeframe = "4h"
leverage = 1.0


def calculate_kama(close, period=14, fast=2, slow=30):
    """
    Kaufman Adaptive Moving Average
    Returns trend direction: 1 = bullish, -1 = bearish, 0 = neutral
    """
    n = len(close)
    if n < period:
        return np.full(n, 0.0)
    
    # Calculate EMA of absolute price change
    delta = np.abs(np.diff(close, prepend=close[0]))
    
    # Efficiency Ratio (ER)
    er = np.full(n, 0.0)
    for i in range(period, n):
        sum_delta = np.sum(delta[i - period + 1:i + 1])
        price_change = np.abs(close[i] - close[i - period])
        if sum_delta > 1e-10:
            er[i] = price_change / sum_delta
    
    # Smoothing constant
    fast_alpha = 2.0 / (fast + 1)
    slow_alpha = 2.0 / (slow + 1)
    
    kama = np.full(n, np.nan, dtype=np.float64)
    kama[period - 1] = close[period - 1]
    
    for i in range(period, n):
        if np.isnan(kama[i - 1]):
            kama[i] = close[i]
        else:
            sc = (er[i] * (fast_alpha - slow_alpha) + slow_alpha) ** 2
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama


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


def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness
    CHOP > 61.8 = ranging (no trades), CHOP < 61.8 = trending (allow trades)
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


def calculate_donchian(high, low, period=20):
    """Donchian Channel - returns upper, middle, lower bands"""
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    middle = (upper + lower) / 2.0
    return upper, middle, lower


def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 12h data for trend bias - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # 12h KAMA for trend direction
    kama_12h_raw = calculate_kama(df_12h['close'].values, period=14)
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h_raw)
    
    # 12h EMA for trend confirmation
    ema_12h_50 = pd.Series(df_12h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_50)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    # Donchian(20) channel
    donchian_upper, donchian_middle, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Volume moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Pre-compute 4h KAMA for intra-bar trend
    kama_4h = calculate_kama(close, period=14)
    ema_4h_21 = pd.Series(close).ewm(span=21, min_periods=21, adjust=False).mean().values
    
    signals = np.zeros(n)
    SIZE = 0.30
    SIZE_HALF = SIZE / 2.0
    
    # Warmup - need at least 20 bars for Donchian
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(chop_14[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(kama_12h_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === TREND BIAS FROM 12h ===
        # KAMA rising = bullish trend
        kama_bullish_12h = kama_12h_raw[-1] > kama_12h_raw[-5] if len(kama_12h_raw) >= 5 else True
        # Use last known 12h KAMA value for current alignment
        kama_12h_val = kama_12h_aligned[i]
        kama_12h_prev = kama_12h_aligned[i - 1] if i > 0 else kama_12h_val
        trend_bullish_12h = kama_12h_val > kama_12h_prev if not (np.isnan(kama_12h_val) or np.isnan(kama_12h_prev)) else True
        
        # Price above 12h EMA = bullish
        price_above_12h_ema = close[i] > ema_12h_aligned[i] if not np.isnan(ema_12h_aligned[i]) else True
        
        # === 4h TREND CONFIRMATION ===
        kama_4h_bullish = kama_4h[i] > kama_4h[i - 1] if i > 0 and not np.isnan(kama_4h[i]) and not np.isnan(kama_4h[i - 1]) else True
        price_above_4h_kama = close[i] > kama_4h[i] if not np.isnan(kama_4h[i]) else True
        
        # === REGIME CHECK (CHOPPINESS) ===
        chop = chop_14[i]
        is_trending = chop < 61.8  # Only trade in trending or neutral markets
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === DONCHIAN BREAKOUT SIGNAL ===
        current_upper = donchian_upper[i]
        current_lower = donchian_lower[i]
        current_middle = donchian_middle[i]
        
        # Previous bar's close for breakout detection
        prev_close = close[i - 1] if i > 0 else close[i]
        
        # Check if price broke above/below yesterday's range
        breakout_up = prev_close < current_upper and close[i] >= current_upper
        breakout_down = prev_close > current_lower and close[i] <= current_lower
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if is_trending:
            # LONG: Breakout above + bullish 12h trend + volume confirm
            if breakout_up:
                # Must have bullish 12h trend or at least not bearish
                if trend_bullish_12h or price_above_12h_ema:
                    # Volume spike is STRONG confirmation
                    if vol_spike:
                        desired_signal = SIZE
                    # Without volume, require 4h trend alignment
                    elif kama_4h_bullish and price_above_4h_kama:
                        desired_signal = SIZE
            
            # SHORT: Breakdown below + bearish 12h trend + volume confirm
            if breakout_down:
                # Must have bearish 12h trend or at least not bullish
                if not trend_bullish_12h or not price_above_12h_ema:
                    # Volume spike is STRONG confirmation
                    if vol_spike:
                        desired_signal = -SIZE
                    # Without volume, require 4h trend alignment
                    elif not kama_4h_bullish and not price_above_4h_kama:
                        desired_signal = -SIZE
        
        signals[i] = desired_signal
    
    return signals