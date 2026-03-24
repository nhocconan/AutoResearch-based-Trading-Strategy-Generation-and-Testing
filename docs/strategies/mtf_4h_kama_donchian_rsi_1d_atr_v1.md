# Strategy: mtf_4h_kama_donchian_rsi_1d_atr_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.565 | -17.2% | -26.3% | 1130 | FAIL |
| ETHUSDT | -0.311 | -10.1% | -27.3% | 1084 | FAIL |
| SOLUSDT | 0.262 | +39.5% | -39.5% | 1162 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.414 | +14.5% | -12.5% | 349 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #1479: 4h Primary + 1d HTF — Simplified Trend Following with KAMA Adaptive Trend

Hypothesis: After 1104 failed strategies, the pattern is clear:
1. Complex regime-switching fails (Choppiness, dual-regime all negative Sharpe)
2. Lower TF (30m, 1h) generates 0 trades due to over-filtering
3. Higher TF (12h, 1d) works better with simpler logic
4. KAMA (Kaufman Adaptive MA) adapts to volatility better than HMA/EMA

Key insight from #1477 (Sharpe=0.150): Simple trend-follow with HTF filter works.
This strategy uses:
- 1d KAMA for macro trend direction (adaptive to volatility)
- 4h KAMA crossover for entry timing
- RSI(14) pullback filter (loose: 40-60 range for more trades)
- Donchian(20) breakout confirmation
- ATR(14)*2.5 trailing stoploss

Why 4h + 1d should work:
1. 4h = target 20-50 trades/year (minimal fee drag ~1-2.5%)
2. 1d KAMA filter prevents trading against macro trend
3. KAMA adapts ER (Efficiency Ratio) - slower in chop, faster in trends
4. Loose RSI filter (40-60 vs 45-55) ensures sufficient trades
5. Discrete signal sizes (0.0, ±0.25, ±0.30) minimize fee churn

Timeframe: 4h
HTF: 1d (call get_htf_data ONCE before loop!)
Position Size: 0.30 (discrete levels)
Target: 20-50 trades/year, Sharpe > 0.150, ALL symbols Sharpe > 0
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_donchian_rsi_1d_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average
    Adapts to market noise using Efficiency Ratio
    Period=10 for ER calculation, fast=2, slow=30 are standard
    """
    n = len(close)
    if n < period + slow_period:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan)
    
    # Calculate Efficiency Ratio (ER)
    er = np.full(n, np.nan)
    for i in range(period, n):
        if np.isnan(close[i]) or np.isnan(close[i - period]):
            continue
        signal = abs(close[i] - close[i - period])
        noise = 0.0
        for j in range(i - period + 1, i + 1):
            if not np.isnan(close[j]) and not np.isnan(close[j-1]):
                noise += abs(close[j] - close[j-1])
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 1.0
    
    # Calculate smoothing constant
    sc = np.full(n, np.nan)
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    for i in range(period, n):
        if not np.isnan(er[i]):
            sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[period] = close[period]
    
    # Calculate KAMA
    for i in range(period + 1, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]) and not np.isnan(close[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    mask = loss_smooth > 1e-10
    rsi[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi[loss_smooth <= 1e-10] = 100.0
    rsi[:period] = np.nan
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Donchian Channel"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.nanmax(high[i - period + 1:i + 1])
        lower[i] = np.nanmin(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_sma(close, period=50):
    """Simple Moving Average for additional trend filter"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d KAMA for macro trend bias
    kama_1d_raw = calculate_kama(df_1d['close'].values, period=10)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    # Calculate primary (4h) indicators
    kama_4h = calculate_kama(close, period=10)
    kama_4h_fast = calculate_kama(close, period=5)  # Faster KAMA for crossover
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    sma_50 = calculate_sma(close, period=50)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):  # Start after indicators are ready
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(kama_1d_aligned[i]) or np.isnan(kama_4h[i]) or np.isnan(sma_50[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === MACRO TREND (1d KAMA) - direction bias ===
        # Only trade in direction of daily trend
        daily_bull = close[i] > kama_1d_aligned[i]
        daily_bear = close[i] < kama_1d_aligned[i]
        
        # === PRIMARY TREND (4h KAMA) ===
        kama_bull = close[i] > kama_4h[i]
        kama_bear = close[i] < kama_4h[i]
        
        # === KAMA CROSSOVER (faster signal) ===
        kama_cross_bull = kama_4h_fast[i] > kama_4h[i]
        kama_cross_bear = kama_4h_fast[i] < kama_4h[i]
        
        # === DONCHIAN BREAKOUT ===
        breakout_high = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        breakout_low = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === RSI MOMENTUM - LOOSE bands for more trades (40-60) ===
        rsi_bullish = rsi[i] > 40.0
        rsi_bearish = rsi[i] < 60.0
        rsi_strong_bull = rsi[i] > 50.0
        rsi_strong_bear = rsi[i] < 50.0
        
        # === SMA 50 FILTER ===
        above_sma50 = close[i] > sma_50[i]
        below_sma50 = close[i] < sma_50[i]
        
        # === DESIRED SIGNAL - SIMPLIFIED TREND FOLLOWING ===
        desired_signal = 0.0
        
        # LONG: Daily bull + 4h bull + Breakout or KAMA cross + RSI support
        if daily_bull and kama_bull:
            if breakout_high and rsi_bullish:
                desired_signal = BASE_SIZE
            elif kama_cross_bull and above_sma50 and rsi_strong_bull:
                desired_signal = BASE_SIZE * 0.8
            elif kama_bull and rsi[i] > 45.0 and close[i] > kama_1d_aligned[i]:
                desired_signal = BASE_SIZE * 0.6  # Weaker signal
        
        # SHORT: Daily bear + 4h bear + Breakout or KAMA cross + RSI support
        elif daily_bear and kama_bear:
            if breakout_low and rsi_bearish:
                desired_signal = -BASE_SIZE
            elif kama_cross_bear and below_sma50 and rsi_strong_bear:
                desired_signal = -BASE_SIZE * 0.8
            elif kama_bear and rsi[i] < 55.0 and close[i] < kama_1d_aligned[i]:
                desired_signal = -BASE_SIZE * 0.6  # Weaker signal
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= BASE_SIZE * 0.7:
            final_signal = BASE_SIZE
        elif desired_signal >= BASE_SIZE * 0.4:
            final_signal = BASE_SIZE * 0.8
        elif desired_signal >= BASE_SIZE * 0.2:
            final_signal = BASE_SIZE * 0.5
        elif desired_signal <= -BASE_SIZE * 0.7:
            final_signal = -BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.4:
            final_signal = -BASE_SIZE * 0.8
        elif desired_signal <= -BASE_SIZE * 0.2:
            final_signal = -BASE_SIZE * 0.5
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals
```

## Last Updated
2026-03-24 00:44
