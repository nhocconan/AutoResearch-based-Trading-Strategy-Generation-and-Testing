#!/usr/bin/env python3
"""
Experiment #1527: 6h Primary + 1d HTF — KAMA Adaptive Trend with RSI Pullback

Hypothesis: 6h timeframe is underexplored (ZERO prior experiments). KAMA (Kaufman 
Adaptive Moving Average) adapts to volatility - fast in trends, slow in ranges.
This is ideal for 6h which sees both trending and ranging periods.

Key components:
1. 1d HMA(21) for major trend bias (avoid counter-trend)
2. 6h KAMA(10,2,30) for adaptive trend following (ER-based smoothing)
3. RSI(14) pullback entries with LOOSE thresholds (35/65 not 30/70)
4. Volume confirmation (taker_buy_volume ratio > 0.52)
5. ATR(14) trailing stoploss at 2.5x

Why this should work:
- KAMA adapts to market regime automatically (no chop index needed)
- LOOSE RSI thresholds (35/65) guarantee sufficient trades
- 1d HMA filter prevents major counter-trend losses
- Volume filter adds confirmation without being too restrictive
- 6h TF = natural 35-55 trades/year (fee-efficient)

Entry logic (LOOSE to guarantee ≥30 trades/train, ≥3/test):
- LONG: 1d_HMA bullish + price>KAMA + RSI(14) 35-55 + volume_confirm
- SHORT: 1d_HMA bearish + price<KAMA + RSI(14) 45-65 + volume_confirm

Target: Sharpe>0.6, trades>=35 train, trades>=4 test, DD>-35%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_kama_adaptive_rsi_volume_1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts smoothing based on market efficiency ratio (ER)
    ER = |close - close_n| / sum(|close_i - close_i-1|)
    SC = [ER * (fast_SC - slow_SC) + slow_SC]^2
    """
    n = len(close)
    if n < period + slow_period:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan, dtype=np.float64)
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n, dtype=np.float64)
    for i in range(period, n):
        price_change = abs(close[i] - close[i - period])
        volatility = 0.0
        for j in range(i - period + 1, i + 1):
            volatility += abs(close[j] - close[j - 1])
        if volatility > 1e-10:
            er[i] = price_change / volatility
    
    # Calculate Smoothing Constant (SC)
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = np.zeros(n, dtype=np.float64)
    for i in range(period, n):
        sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[period] = close[period]
    
    # Calculate KAMA
    for i in range(period + 1, n):
        if not np.isnan(kama[i - 1]):
            kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while smoothing"""
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

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain = np.insert(gain, 0, 0)
    loss = np.insert(loss, 0, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    mask = avg_loss != 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_vol = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 6h indicators
    kama_10 = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    
    # Volume ratio (taker buy / total volume)
    vol_ratio = np.zeros(n, dtype=np.float64)
    mask = volume > 1e-10
    vol_ratio[mask] = taker_buy_vol[mask] / volume[mask]
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup period
    min_bars = 80
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(kama_10[i]):
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
        
        # === TREND DIRECTION (1d HMA bias) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # === KAMA TREND (adaptive) ===
        price_above_kama = close[i] > kama_10[i]
        price_below_kama = close[i] < kama_10[i]
        
        # === RSI ===
        rsi = rsi_14[i]
        
        # === VOLUME CONFIRMATION ===
        vol_ratio_val = vol_ratio[i]
        vol_bullish = vol_ratio_val > 0.52  # More buyer pressure
        vol_bearish = vol_ratio_val < 0.48  # More seller pressure
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # LONG: 1d bullish + price above KAMA + RSI pullback (35-55) + volume confirm
        if price_above_1d and price_above_kama:
            if 35 <= rsi <= 55 and vol_bullish:
                desired_signal = SIZE_STRONG
            elif 30 <= rsi < 35:  # Deeper pullback = strong signal
                desired_signal = SIZE_STRONG
            elif 55 < rsi <= 60 and vol_bullish:  # Momentum continuation
                desired_signal = SIZE_BASE
        
        # SHORT: 1d bearish + price below KAMA + RSI bounce (45-65) + volume confirm
        elif price_below_1d and price_below_kama:
            if 45 <= rsi <= 65 and vol_bearish:
                desired_signal = -SIZE_STRONG
            elif 65 < rsi <= 70:  # Higher overbought = strong signal
                desired_signal = -SIZE_STRONG
            elif 40 <= rsi < 45 and vol_bearish:  # Momentum continuation
                desired_signal = -SIZE_BASE
        
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