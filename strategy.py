#!/usr/bin/env python3
"""
Experiment #010: 1d KAMA Adaptive Trend + RSI Extremes + Weekly Regime

HYPOTHESIS: KAMA (Kaufman Adaptive Moving Average) adapts to market noise,
reducing whipsaws in choppy periods while capturing trends efficiently.
RSI extremes identify overbought/oversold conditions for entry timing.
Weekly HMA provides macro trend bias. Choppiness filter avoids range markets.

WHY THIS SHOULD WORK IN BOTH BULL AND BEAR:
- Bull markets: KAMA turns up + RSI from oversold + price > weekly HMA → long
- Bear markets: KAMA turns down + RSI from overbought + price < weekly HMA → short
- Range markets: Choppiness filter keeps us out (CHOP > 61.8)

TARGET: 60-120 total trades over 4 years (15-30/year) for 1d timeframe.
DB reference: mtf_1d_kama_rsi_chop_regime_1w_v1 (Sharpe=1.310, 74 trades)
This version loosens RSI thresholds slightly to ensure >=50 trades minimum.

KEY DESIGN:
1. KAMA(10) direction change as primary trend signal
2. RSI(14) extremes (25/75) for entry timing
3. 1w HMA(21) for macro trend bias
4. Choppiness < 58 for trending regime (slightly looser)
5. ATR(14) stoploss at 2.5x
6. Signal: 0.30 (discrete)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_rsi_chop_1w_v3"
timeframe = "1d"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average
    Adapts smoothing based on market efficiency ratio
    """
    n = len(close)
    kama = np.full(n, np.nan, dtype=np.float64)
    
    if n < er_period + slow_period:
        return kama
    
    # Calculate Efficiency Ratio (ER)
    er = np.full(n, np.nan, dtype=np.float64)
    for i in range(er_period, n):
        if not np.isnan(close[i]) and not np.isnan(close[i - er_period]):
            signal = abs(close[i] - close[i - er_period])
            noise = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
            if noise > 1e-10:
                er[i] = signal / noise
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        if np.isnan(er[i]):
            continue
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if avg_loss[i] == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
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
    """
    Choppiness Index - measures market choppiness
    CHOP > 61.8 = ranging, CHOP < 38.2 = trending
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1w data for trend bias
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA for trend
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 1d indicators
    kama_10 = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    rsi_14 = calculate_rsi(close, period=14)
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
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Track KAMA direction
    kama_prev = np.full(n, np.nan, dtype=np.float64)
    for i in range(1, n):
        if not np.isnan(kama_10[i]) and not np.isnan(kama_10[i-1]):
            kama_prev[i] = kama_10[i-1]
    
    # Warmup
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1w_aligned[i]) or np.isnan(kama_10[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME CHECK ===
        chop = chop_14[i]
        is_trending = chop < 58.0  # Slightly looser to allow more trades
        
        # === TREND BIAS (1w HMA) ===
        price_above_1w_hma = close[i] > hma_1w_aligned[i]
        
        # === KAMA DIRECTION ===
        kama_bullish = kama_10[i] > kama_prev[i] if not np.isnan(kama_prev[i]) else False
        kama_bearish = kama_10[i] < kama_prev[i] if not np.isnan(kama_prev[i]) else False
        
        # === RSI EXTREMES (looser thresholds for more trades) ===
        rsi = rsi_14[i]
        rsi_oversold = rsi < 35.0  # Looser than 25
        rsi_overbought = rsi > 65.0  # Looser than 75
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.3  # Slightly looser
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: KAMA turns up + RSI from oversold + bullish weekly trend
        if is_trending and kama_bullish and rsi_oversold:
            if price_above_1w_hma:
                if vol_spike:
                    desired_signal = SIZE
                else:
                    desired_signal = SIZE * 0.7  # Enter anyway, smaller size
        
        # SHORT: KAMA turns down + RSI from overbought + bearish weekly trend
        if is_trending and kama_bearish and rsi_overbought:
            if not price_above_1w_hma:
                if vol_spike:
                    desired_signal = -SIZE
                else:
                    desired_signal = -SIZE * 0.7  # Enter anyway, smaller size
        
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
        
        # === EXIT ON OPPOSITE SIGNAL ===
        if in_position and position_side > 0 and desired_signal < 0:
            desired_signal = 0.0
        if in_position and position_side < 0 and desired_signal > 0:
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