# Strategy: mtf_1d_kama_trend_1w_hma_rsi_atr_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.135 | +9.8% | -15.2% | 119 | FAIL |
| ETHUSDT | -0.396 | -10.7% | -38.7% | 124 | FAIL |
| SOLUSDT | 1.036 | +210.2% | -26.7% | 124 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.172 | +8.1% | -10.6% | 26 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #1557: 1d Primary + 1w HTF — KAMA Adaptive Trend Strategy

Hypothesis: After 1157 failed experiments, key insights:
1. 1d timeframe produces best Sharpe ratios (current best: 0.618)
2. KAMA (Kaufman Adaptive Moving Average) outperforms HMA/EMA in crypto
   - Adapts to volatility: fast in trends, slow in ranges
   - Reduces whipsaw during 2022 crash and 2025 bear market
3. Single HTF filter (1w HMA) avoids conflicting signals
4. Loose RSI thresholds ensure 20-50 trades/year target
5. ATR trailing stop protects from major drawdowns

Strategy Design:
- HTF Bias: 1w HMA(21) for macro trend direction ONLY
- Primary: 1d KAMA(10,2,30) for adaptive trend following
- Entry: Price crosses KAMA + RSI(14) confirmation (>40 long, <60 short)
- Exit: 2.5x ATR(14) trailing stop via signal→0
- Size: 0.30 discrete (0.0, ±0.30) to minimize fee churn

Why KAMA for crypto:
- Efficiency Ratio adapts smoothing constant based on trend strength
- During strong trends (2021 bull, SOL rallies): KAMA follows closely
- During chop (2022 crash, 2025 bear): KAMA flattens, reduces whipsaw
- Proven in literature for noisy markets like crypto

Timeframe: 1d (required for this experiment)
HTF: 1w HMA(21) for bias only
Target: Sharpe > 0.618, trades > 20/train, > 3/test, DD < -35%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_trend_1w_hma_rsi_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    
    Adapts to market volatility using Efficiency Ratio (ER).
    ER = |Net Change| / Sum of Absolute Changes over period
    SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
    
    When trending (ER high): SC approaches fast_sc, KAMA follows price
    When ranging (ER low): SC approaches slow_sc, KAMA flattens
    
    Standard parameters: er_period=10, fast_period=2, slow_period=30
    """
    n = len(close)
    if n < slow_period + er_period:
        return np.full(n, np.nan)
    
    # Calculate Efficiency Ratio
    er = np.full(n, np.nan)
    for i in range(er_period - 1, n):
        net_change = abs(close[i] - close[i - er_period + 1])
        sum_changes = np.sum(np.abs(np.diff(close[i - er_period + 1:i + 1])))
        if sum_changes > 1e-10:
            er[i] = net_change / sum_changes
        else:
            er[i] = 0.0
    
    # Calculate smoothing constants
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    # Calculate KAMA
    kama = np.full(n, np.nan)
    kama[er_period - 1] = close[er_period - 1]  # Initialize with price
    
    for i in range(er_period, n):
        if np.isnan(er[i]):
            continue
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = max(int(np.sqrt(period)), 1)
    
    def wma(data, w_period):
        result = np.full(len(data), np.nan)
        if w_period < 1:
            return result
        weights = np.arange(1, w_period + 1, dtype=float)
        weights /= weights.sum()
        for i in range(w_period - 1, len(data)):
            window = data[i - w_period + 1:i + 1]
            if np.any(np.isnan(window)):
                continue
            result[i] = np.sum(window * weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan)
    mask = ~np.isnan(wma_half) & ~np.isnan(wma_full)
    diff[mask] = 2.0 * wma_half[mask] - wma_full[mask]
    
    hma = wma(diff, sqrt_period)
    return hma

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for macro trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    rsi_14 = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    # KAMA for adaptive trend following
    kama = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1w_aligned[i]) or np.isnan(kama[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === MACRO TREND BIAS (1w HMA) ===
        weekly_bull = close[i] > hma_1w_aligned[i]
        weekly_bear = close[i] < hma_1w_aligned[i]
        
        # === PRIMARY TREND (1d KAMA) ===
        kama_bull = close[i] > kama[i]
        kama_bear = close[i] < kama[i]
        
        # === RSI FILTER (LOOSE — ensures trades fire) ===
        rsi_long_ok = rsi_14[i] > 40.0
        rsi_short_ok = rsi_14[i] < 60.0
        
        # === ENTRY LOGIC — SIMPLE & LOOSE ===
        desired_signal = 0.0
        
        # LONG: Weekly bull + Daily KAMA bull + RSI confirmation
        if weekly_bull and kama_bull and rsi_long_ok:
            desired_signal = BASE_SIZE
        
        # SHORT: Weekly bear + Daily KAMA bear + RSI confirmation
        if weekly_bear and kama_bear and rsi_short_ok:
            desired_signal = -BASE_SIZE
        
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
        if desired_signal >= BASE_SIZE * 0.85:
            final_signal = BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.85:
            final_signal = -BASE_SIZE
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
2026-03-24 01:43
