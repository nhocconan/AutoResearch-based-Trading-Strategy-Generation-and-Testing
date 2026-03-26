#!/usr/bin/env python3
"""
Experiment #010: 1d Primary + 1w HTF — Simple KAMA Trend

Hypothesis: 1d timeframe with 1w trend bias using KAMA direction.
Simple entry: KAMA direction + RSI confirmation (no multiple filters).
Donchian breakout for structure. Choppiness for regime avoidance.

Why it should work in BOTH bull AND bear:
- Bull: KAMA up + RSI confirms = long
- Bear: KAMA down + RSI confirms = short
- Range: Choppiness filter avoids whipsaws
- 1d = fewer trades = less fee drag

Key design choices:
1. KAMA(10) for trend - responsive yet smooth
2. RSI(14) as ONLY confirmation - no stacking
3. Donchian(20) for structure breakout
4. Choppiness(14) to avoid range markets
5. 1w HMA(21) for HTF trend bias
6. Simple 2-condition entries = guaranteed trades
7. 0.30 signal size, 2.5x ATR stoploss

Target: Sharpe>0.5, trades 75-200 total (19-50/year), DD>-40%
Timeframe: 1d
Size: 0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_rsi_donchian_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Kaufman Adaptive Moving Average
    ERA = direction / volatility
    """
    n = len(close)
    if n < period + slow:
        return np.full(n, np.nan)
    
    # Price change
    change = np.abs(np.diff(close, prepend=close[0]))
    
    # Volatility (sum of price changes over period)
    volatility = np.zeros(n, dtype=np.float64)
    for i in range(period, n):
        volatility[i] = np.sum(change[i - period + 1:i + 1])
    
    # Efficiency Ratio (ERA)
    era = np.zeros(n, dtype=np.float64)
    mask = volatility > 1e-10
    era[mask] = change[mask] / volatility[mask]
    era = np.clip(era, 0, 1)
    
    # Smoothing constant
    fast_const = 2 / (fast + 1)
    slow_const = 2 / (slow + 1)
    
    kama = np.full(n, np.nan, dtype=np.float64)
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        sc = (era[i] * (fast_const - slow_const) + slow_const) ** 2
        kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    return kama

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    mask = avg_loss > 1e-10
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    
    return rsi

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
    """Choppiness Index - CHOP > 61.8 = ranging, CHOP < 38.2 = trending"""
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
    """Donchian Channel - price channel breakout structure"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

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

def calculate_volume SMA(close, volume, period=20):
    """Volume SMA for confirmation"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA for trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 1d indicators
    kama_10 = calculate_kama(close, period=10)
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    vol_sma_20 = calculate_volumeSMA(close, volume, period=20)
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    stop_price = 0.0
    
    # Warmup period
    min_bars = 50
    
    for i in range(min_bars, n):
        # Check indicators ready
        if np.isnan(kama_10[i]) or np.isnan(rsi_14[i]) or np.isnan(atr_14[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === REGIME CHECK ===
        chop = chop_14[i] if not np.isnan(chop_14[i]) else 50.0
        is_trending = chop < 50.0  # Trend regime
        
        # === KAMA TREND DIRECTION ===
        kama_trend_up = close[i] > kama_10[i]
        kama_trend_down = close[i] < kama_10[i]
        
        # === 1w HMA TREND BIAS ===
        hma_1w_trend_up = hma_1w_aligned[i] > hma_1w_aligned[i-1] if i > 0 and not np.isnan(hma_1w_aligned[i-1]) else True
        
        # === RSI CONFIRMATION (not extreme = healthy) ===
        rsi_val = rsi_14[i]
        rsi_healthy_long = 35 < rsi_val < 65  # Not overbought, not oversold
        rsi_healthy_short = 35 < rsi_val < 65
        
        # === DONCHIAN BREAKOUT ===
        # Long: price breaks above 20d high
        # Short: price breaks below 20d low
        donch_break_long = close[i] > donch_upper[i-1] if i > 0 and not np.isnan(donch_upper[i-1]) else False
        donch_break_short = close[i] < donch_lower[i-1] if i > 0 and not np.isnan(donch_lower[i-1]) else False
        
        # === VOLUME CONFIRMATION ===
        vol_confirm = volume[i] > vol_sma_20[i] if not np.isnan(vol_sma_20[i]) else True
        
        # === SIMPLE 2-CONDITION ENTRY ===
        # Entry requires: KAMA direction + RSI healthy + (Donchian breakout OR strong trend)
        desired_signal = 0.0
        
        if is_trending:
            # TREND REGIME: KAMA + RSI + volume
            # LONG: KAMA up + RSI healthy + volume up
            if kama_trend_up and rsi_healthy_long and vol_confirm:
                desired_signal = SIZE
            
            # SHORT: KAMA down + RSI healthy + volume up
            elif kama_trend_down and rsi_healthy_short and vol_confirm:
                desired_signal = -SIZE
        
        else:
            # RANGE REGIME: Donchian breakout more reliable
            # Only enter on clear breakout with volume
            if donch_break_long and kama_trend_up and vol_confirm:
                desired_signal = SIZE
            elif donch_break_short and kama_trend_down and vol_confirm:
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
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
        
        # === DISCRETIZE ===
        if abs(desired_signal) >= SIZE * 0.8:
            final_signal = SIZE if desired_signal > 0 else -SIZE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                # New entry
                in_position = True
                position_side = int(np.sign(final_signal))
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
        
        signals[i] = final_signal
    
    return signals