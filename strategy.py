#!/usr/bin/env python3
"""
Experiment #001: SIMPLIFIED Dual-Regime Breakout Strategy

Hypothesis: Most failures are due to TOO MANY overlapping conditions.
The winning formula from DB is SIMPLE: ONE strong entry + volume + regime.

Key design (learned from 17 failures):
1. SINGLE entry type per regime (not 4+ stacked conditions)
2. TREND regime: Donchian breakout only (proven pattern from DB winners)
3. RANGE regime: Bollinger squeeze only (mean reversion edge)
4. 1d HMA bias as primary filter (clear trend direction)
5. Volume confirmation (simple: above/below 20d MA)
6. Choppiness regime filter (no trend following in chop)
7. 2x ATR trailing stoploss

Why this should work in BOTH bull AND bear:
- Bull (2020-2021): Donchian breakout catches momentum
- Bear (2022): Bollinger squeeze mean reversion catches bounces
- Range (2023-2024): Choppiness filter avoids whipsaws

Target: Sharpe>0.5, trades 75-200 total/4yr, DD>-35%
Timeframe: 4h | Size: 0.25/0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_simple_dual_regime_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - smoother with less lag"""
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures if market is trending or ranging
    CHOP > 61.8 = ranging (mean reversion), CHOP < 38.2 = trending (trend following)
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
    """Donchian Channel - price channel breakout"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_bollinger_squeeze(close, high, low, period=20, std_mult=2.0):
    """
    Bollinger Squeeze detection - identifies low-volatility compression
    Returns squeeze_on (bool) and band position
    """
    n = len(close)
    if n < period:
        return np.full(n, False), np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    # ATR-based Keltner Channel for comparison
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    kelt_upper = sma + atr * 2.0
    kelt_lower = sma - atr * 2.0
    
    # Squeeze = BB inside Keltner
    squeeze_on = np.full(n, False, dtype=bool)
    for i in range(period, n):
        if upper[i] < kelt_upper[i] and lower[i] > kelt_lower[i]:
            squeeze_on[i] = True
    
    # Band position: 0=lower, 0.5=middle, 1=upper
    band_pos = np.full(n, 0.5, dtype=np.float64)
    band_range = upper - lower
    mask = band_range > 1e-10
    band_pos[mask] = (close[mask] - lower[mask]) / band_range[mask]
    
    return squeeze_on, upper, sma, lower

def calculate_volume_ma(volume, period=20):
    """Simple volume moving average for confirmation"""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    squeeze_on, bb_upper, bb_mid, bb_lower = calculate_bollinger_squeeze(close, high, low, period=20)
    volume_ma = calculate_volume_ma(volume, period=20)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup period
    min_bars = 50
    
    for i in range(min_bars, n):
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
        
        if np.isnan(bb_lower[i]) or np.isnan(donch_upper[i]):
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
        
        # === REGIME DETECTION ===
        chop = chop_14[i]
        is_trend_regime = chop < 38.2
        is_range_regime = chop > 61.8
        
        # === 1d TREND BIAS ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # === VOLUME CONFIRMATION (simple: above MA) ===
        volume_above_ma = volume[i] > volume_ma[i] if not np.isnan(volume_ma[i]) else True
        
        # === DONCHIAN BREAKOUT (for trend regime) ===
        donch_breakout_long = False
        donch_breakout_short = False
        
        if i > 20 and not np.isnan(donch_upper[i-1]) and not np.isnan(donch_lower[i-1]):
            # Price breaks above 20-day high
            donch_breakout_long = close[i] > donch_upper[i-1] and close[i-1] <= donch_upper[i-1]
            # Price breaks below 20-day low
            donch_breakout_short = close[i] < donch_lower[i-1] and close[i-1] >= donch_lower[i-1]
        
        # === BOLLINGER SQUEEZE (for range regime) ===
        # Check if price bounced from lower band during squeeze
        squeeze_bounce_long = False
        squeeze_bounce_short = False
        
        if squeeze_on[i-1] and not squeeze_on[i]:  # Squeeze just fired
            if close[i] > bb_mid[i]:  # Price moved above midpoint after squeeze
                squeeze_bounce_long = True
            elif close[i] < bb_mid[i]:  # Price moved below midpoint after squeeze
                squeeze_bounce_short = True
        
        # Alternative: mean reversion from bands during range
        bb_touch_lower = close[i] <= bb_lower[i] * 1.01
        bb_touch_upper = close[i] >= bb_upper[i] * 0.99
        
        # === SIMPLE ENTRY LOGIC (ONE primary condition per regime) ===
        desired_signal = 0.0
        
        # TREND REGIME: Donchian breakout + 1d bias + volume
        if is_trend_regime:
            # LONG: 1d bullish + Donchian breakout up + volume confirmation
            if price_above_1d and donch_breakout_long and volume_above_ma:
                desired_signal = SIZE_STRONG
            
            # SHORT: 1d bearish + Donchian breakout down + volume confirmation
            elif price_below_1d and donch_breakout_short and volume_above_ma:
                desired_signal = -SIZE_STRONG
        
        # RANGE REGIME: Mean reversion from Bollinger bands
        elif is_range_regime:
            # LONG: Price at lower BB + 1d bullish bias (testing support)
            if price_above_1d and bb_touch_lower:
                desired_signal = SIZE_BASE
            
            # SHORT: Price at upper BB + 1d bearish bias (testing resistance)
            elif price_below_1d and bb_touch_upper:
                desired_signal = -SIZE_BASE
        
        # NEUTRAL REGIME: Skip (no clear edge)
        # This is intentional - we only trade in TREND or RANGE regimes
        
        # === STOPLOSS CHECK (2x ATR trailing) ===
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
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
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
        
        signals[i] = final_signal
    
    return signals