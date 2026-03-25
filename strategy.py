#!/usr/bin/env python3
"""
Experiment #1642: 4h Primary + 1d HTF — KAMA Trend + RSI Pullback (LOOSE)

Hypothesis: Simplified 4h strategy with KAMA adaptive trend + RSI pullback entries
will generate sufficient trades while maintaining edge. Key learnings from failures:

1. LOOSE RSI thresholds (30/70 not 35/65) — guarantees entries on pullbacks
2. NO Choppiness filter — was too restrictive (#1618, #1639 failures)
3. Simple 1d HMA(21) trend bias — proven in #1638 (Sharpe=0.014, +46.9%)
4. KAMA(10) adaptive smoothing — better than EMA in crypto volatility
5. ATR 2.5x trailing stop — protects from 2022-style crashes
6. Discrete sizes 0.25/0.30 — minimizes fee churn

Why this beats #1638 (mtf_4h_hma_rsi_pullback_1d_loose_v1):
- KAMA adapts to volatility (ER-based) vs static HMA
- RSI 30/70 extremes = more entry opportunities than 35/65
- Donchian breakout confirmation = catches momentum continuations
- Simpler logic = fewer conditions that can all fail simultaneously

Target: Sharpe>0.6, trades≥40 train, trades≥5 test, DD>-35%
Timeframe: 4h
Size: 0.25 base, 0.30 strong (discrete)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_rsi_pullback_1d_loose_v2"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts smoothing based on market efficiency ratio
    ER=1.0 (trending) → fast EMA, ER=0.0 (choppy) → slow SMA
    """
    n = len(close)
    if n < er_period + slow_period:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan, dtype=np.float64)
    
    # Calculate Efficiency Ratio (ER)
    er = np.full(n, np.nan, dtype=np.float64)
    for i in range(er_period, n):
        price_change = abs(close[i] - close[i - er_period])
        if price_change < 1e-10:
            er[i] = 0.0
        else:
            vol_sum = 0.0
            for j in range(i - er_period + 1, i + 1):
                vol_sum += abs(close[j] - close[j - 1])
            if vol_sum > 1e-10:
                er[i] = price_change / vol_sum
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        if np.isnan(er[i]):
            kama[i] = kama[i - 1]
        else:
            sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - breakout levels"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_keltner(high, low, close, atr_period=14, mult=2.0):
    """Keltner Channels - volatility-based bands"""
    n = len(close)
    if n < atr_period + 20:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    ema = pd.Series(close).ewm(span=20, min_periods=20, adjust=False).mean().values
    atr = calculate_atr(high, low, close, atr_period)
    
    upper = ema + mult * atr
    lower = ema - mult * atr
    
    return upper, ema, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 4h indicators
    kama_10 = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    kelt_upper, kelt_mid, kelt_lower = calculate_keltner(high, low, close, atr_period=14, mult=2.0)
    
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
    min_bars = 60
    
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
        
        if np.isnan(donch_upper[i]) or np.isnan(kelt_lower[i]):
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
        
        # === KAMA TREND (4h adaptive) ===
        kama_bullish = close[i] > kama_10[i]
        kama_bearish = close[i] < kama_10[i]
        
        # === RSI PULLBACK (LOOSE thresholds for trades) ===
        rsi_val = rsi_14[i]
        rsi_oversold = rsi_val < 30  # LOOSE - was 35
        rsi_overbought = rsi_val > 70  # LOOSE - was 65
        rsi_neutral_bull = 35 < rsi_val < 55  # pullback zone in uptrend
        rsi_neutral_bear = 45 < rsi_val < 65  # pullback zone in downtrend
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = False
        donchian_breakout_short = False
        if i > 0 and not np.isnan(donch_upper[i-1]) and not np.isnan(donch_lower[i-1]):
            donchian_breakout_long = close[i] > donch_upper[i-1]
            donchian_breakout_short = close[i] < donch_lower[i-1]
        
        # === KELTNER TOUCH (volatility extremes) ===
        kelt_touch_lower = close[i] <= kelt_lower[i] * 1.01
        kelt_touch_upper = close[i] >= kelt_upper[i] * 0.99
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # LONG ENTRIES (multiple paths to ensure trades)
        if price_above_1d:  # 1d trend bullish
            # Path 1: KAMA bullish + RSI pullback (main entry)
            if kama_bullish and rsi_neutral_bull:
                desired_signal = SIZE_STRONG if donchian_breakout_long else SIZE_BASE
            
            # Path 2: RSI oversold bounce (mean reversion in uptrend)
            elif kama_bullish and rsi_oversold:
                desired_signal = SIZE_BASE
            
            # Path 3: Keltner lower touch + RSI not overbought
            elif kelt_touch_lower and rsi_val < 60:
                desired_signal = SIZE_BASE
        
        # SHORT ENTRIES (multiple paths to ensure trades)
        elif price_below_1d:  # 1d trend bearish
            # Path 1: KAMA bearish + RSI pullback (main entry)
            if kama_bearish and rsi_neutral_bear:
                desired_signal = -SIZE_STRONG if donchian_breakout_short else -SIZE_BASE
            
            # Path 2: RSI overbought rejection (mean reversion in downtrend)
            elif kama_bearish and rsi_overbought:
                desired_signal = -SIZE_BASE
            
            # Path 3: Keltner upper touch + RSI not oversold
            elif kelt_touch_upper and rsi_val > 40:
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