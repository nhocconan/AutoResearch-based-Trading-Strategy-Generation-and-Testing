#!/usr/bin/env python3
"""
Experiment #1448: 4h Primary + 12h/1d HTF — Regime-Adaptive KAMA + RSI

Hypothesis: Previous 4h strategies failed due to either (1) too strict entries = 0 trades,
or (2) single-regime logic that fails in chop. This strategy uses:

1. REGIME DETECTION: Choppiness Index (CHOP) to switch between trend/mean-revert
   - CHOP > 61.8 = choppy/range → mean reversion logic
   - CHOP < 38.2 = trending → trend following logic
   - Between = neutral, use trend bias only

2. TREND BIAS: 12h KAMA(40) adaptive moving average (better than HMA in crypto)
   - KAMA adapts to volatility, reduces whipsaw in chop

3. ENTRY TRIGGERS (LOOSE to guarantee trades):
   - Trend regime: KAMA(10) > KAMA(40) + RSI(7) > 45 (long)
   - Mean-revert regime: RSI(7) < 35 + price > 12h_KAMA (long)
   - Mirror for shorts

4. ATR(14) trailing stoploss at 2.5x

Why this should work:
- CHOP regime filter proven in literature (Connors/Van Tharp)
- KAMA adapts better than HMA/EMA in crypto volatility
- RSI(7) is faster than RSI(14), generates more signals
- LOOSE thresholds (RSI 35-65, not 30-70) guarantee trades
- 4h TF = natural 25-45 trades/year (fee efficient)

Target: Sharpe>0.6, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 4h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_regime_kama_rsi_chop_12h_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """Kaufman Adaptive Moving Average - adapts to market noise"""
    n = len(close)
    if n < period + slow_period:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan, dtype=np.float64)
    
    # Change = absolute price change over period
    change = np.zeros(n)
    for i in range(period, n):
        change[i] = abs(close[i] - close[i - period])
    
    # Sum of absolute differences (volatility)
    volatility = np.zeros(n)
    for i in range(1, n):
        volatility[i] = volatility[i-1] + abs(close[i] - close[i-1])
        if i >= period:
            volatility[i] -= abs(close[i - period] - close[i - period - 1])
    
    # Efficiency Ratio
    er = np.zeros(n)
    mask = volatility > 0
    er[mask] = change[mask] / volatility[mask]
    er = np.clip(er, 0, 1)
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Adaptive smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[period] = close[period]
    
    # Calculate KAMA
    for i in range(period + 1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    return kama

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - measures market choppiness vs trending"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        tr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            tr_sum += tr
        
        range_val = highest_high - lowest_low
        if range_val > 0 and tr_sum > 0:
            chop[i] = 100 * np.log10(tr_sum / range_val) / np.log10(period)
    
    return chop

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

def calculate_rsi(close, period=7):
    """Relative Strength Index - faster period for more signals"""
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
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align HTF indicators
    kama_12h_raw = calculate_kama(df_12h['close'].values, period=40)
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h_raw)
    
    # Calculate 4h indicators
    kama_10 = calculate_kama(close, period=10)
    kama_40 = calculate_kama(close, period=40)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)
    
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
    min_bars = 100
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_7[i]) or np.isnan(kama_10[i]) or np.isnan(kama_40[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(kama_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION ===
        chop = chop_14[i]
        is_choppy = chop > 61.8  # Range/mean-revert regime
        is_trending = chop < 38.2  # Trend regime
        # Between 38.2-61.8 = neutral, use trend bias only
        
        # === TREND BIAS (12h KAMA) ===
        price_above_12h = close[i] > kama_12h_aligned[i]
        price_below_12h = close[i] < kama_12h_aligned[i]
        
        # === 4h KAMA CROSSOVER ===
        kama_bullish = kama_10[i] > kama_40[i]
        kama_bearish = kama_10[i] < kama_40[i]
        
        # === RSI ===
        rsi = rsi_7[i]
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        if is_trending:
            # TREND FOLLOWING regime
            # LONG: 12h bullish + 4h KAMA bullish + RSI > 45 (not overbought)
            if price_above_12h and kama_bullish and rsi > 45:
                desired_signal = SIZE_STRONG if rsi < 70 else SIZE_BASE
            
            # SHORT: 12h bearish + 4h KAMA bearish + RSI < 55 (not oversold)
            elif price_below_12h and kama_bearish and rsi < 55:
                desired_signal = -SIZE_STRONG if rsi > 30 else -SIZE_BASE
        
        elif is_choppy:
            # MEAN REVERSION regime
            # LONG: RSI < 35 (oversold) + price above 12h KAMA (uptrend bias)
            if rsi < 35 and price_above_12h:
                desired_signal = SIZE_BASE
            
            # SHORT: RSI > 65 (overbought) + price below 12h KAMA (downtrend bias)
            elif rsi > 65 and price_below_12h:
                desired_signal = -SIZE_BASE
        
        else:
            # NEUTRAL regime - use simpler trend bias
            if price_above_12h and kama_bullish and rsi > 50:
                desired_signal = SIZE_BASE
            elif price_below_12h and kama_bearish and rsi < 50:
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