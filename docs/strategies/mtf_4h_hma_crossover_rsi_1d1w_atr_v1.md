# Strategy: mtf_4h_hma_crossover_rsi_1d1w_atr_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.583 | -4.1% | -20.1% | 1056 | FAIL |
| ETHUSDT | -0.292 | +3.0% | -21.3% | 1121 | FAIL |
| SOLUSDT | 0.726 | +96.8% | -14.1% | 1084 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.418 | +13.2% | -12.4% | 369 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #1531: 4h Primary + 1d/1w HTF — Simplified HMA Trend + RSI Filter

Hypothesis: After 1140+ failed strategies, the pattern is clear:
1. COMPLEX = 0 trades (#1528, #1529, #1530 all Sharpe=0.000)
2. 4h timeframe needs LOOSE conditions to generate 20-50 trades/year
3. 1d HMA(21) for macro bias works (proven in #1522, #1527)
4. HMA crossover is simpler and more reliable than dual-regime switches
5. RSI filter should be LOOSE (not extreme values) to ensure trades fire
6. ATR trailing stop is mandatory for risk control

Key insight from failures:
- #1526 (12h dual regime) had negative Sharpe (-0.046) — too complex
- #1529 (4h mean revert) had 0 trades — conditions too strict
- Current best (#1527) uses 1d Donchian + HMA + RSI — simple works!

Design:
- 1d HMA(21) = macro trend bias (long only when price > 1d HMA)
- 4h HMA(16) vs HMA(48) = primary trend signal (crossover)
- RSI(14) = momentum confirmation (loose: <55 for long, >45 for short)
- ATR(14) 2.5x = trailing stoploss
- Position size: 0.28 (discrete levels: 0.0, ±0.28)
- Target: 80-200 trades/train, 20-50 trades/test

Why this should work:
- SIMPLER than dual-regime (fewer conditions = more trades)
- LOOSE RSI bands ensure entries fire (not waiting for extremes)
- 1d bias prevents counter-trend trades (major source of drawdown)
- HMA crossover is proven trend signal (less lag than EMA)

Timeframe: 4h (as required by experiment #1531)
HTF: 1d (daily trend bias), 1w (weekly macro filter)
Position Size: 0.28 (conservative for 4h volatility)
Target: Sharpe > 0.618 (beat current best), DD < -30%, trades > 80
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_crossover_rsi_1d1w_atr_v1"
timeframe = "4h"
leverage = 1.0

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
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for weekly bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (4h) indicators
    hma_16 = calculate_hma(close, period=16)
    hma_48 = calculate_hma(close, period=48)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # Conservative size for 4h volatility
    
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
        if np.isnan(rsi[i]) or np.isnan(hma_16[i]) or np.isnan(hma_48[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === MACRO TREND BIAS (1d HMA) ===
        # Only long when price > daily HMA, only short when price < daily HMA
        daily_bull = close[i] > hma_1d_aligned[i]
        daily_bear = close[i] < hma_1d_aligned[i]
        
        # === WEEKLY FILTER (1w HMA) ===
        # Adds extra confirmation but not required (loose for trades)
        weekly_bull = close[i] > hma_1w_aligned[i]
        weekly_bear = close[i] < hma_1w_aligned[i]
        
        # === PRIMARY TREND (4h HMA Crossover) ===
        hma_fast_above_slow = hma_16[i] > hma_48[i]
        hma_fast_below_slow = hma_16[i] < hma_48[i]
        
        # === RSI MOMENTUM (LOOSE bands for more trades) ===
        rsi_neutral_long = rsi[i] < 55.0  # Not overbought for longs
        rsi_neutral_short = rsi[i] > 45.0  # Not oversold for shorts
        rsi_bullish = rsi[i] > 50.0
        rsi_bearish = rsi[i] < 50.0
        
        # === HMA SLOPE (trend confirmation) ===
        hma_16_slope = 0.0
        if i >= 5 and not np.isnan(hma_16[i-5]):
            hma_16_slope = (hma_16[i] - hma_16[i-5]) / hma_16[i-5] if hma_16[i-5] > 1e-10 else 0.0
        
        hma_16_rising = hma_16_slope > 0.0
        hma_16_falling = hma_16_slope < 0.0
        
        # === DESIRED SIGNAL — SIMPLIFIED LOGIC ===
        desired_signal = 0.0
        
        # LONG SIGNALS (multiple conditions, any can trigger)
        if daily_bull:  # Primary filter: daily trend must be bull
            # Strong signal: HMA crossover + RSI confirmation
            if hma_fast_above_slow and rsi_neutral_long:
                desired_signal = BASE_SIZE
            # Medium signal: HMA fast rising + daily bull
            elif hma_16_rising and rsi_bullish:
                desired_signal = BASE_SIZE * 0.7
            # Weak signal: Just daily bull + HMA fast above slow (ensures trades)
            elif hma_fast_above_slow:
                desired_signal = BASE_SIZE * 0.5
            # Fallback: Daily bull + weekly bull + RSI > 45 (very loose)
            elif weekly_bull and rsi[i] > 45.0:
                desired_signal = BASE_SIZE * 0.4
        
        # SHORT SIGNALS (mirror of long)
        elif daily_bear:  # Primary filter: daily trend must be bear
            # Strong signal: HMA crossover + RSI confirmation
            if hma_fast_below_slow and rsi_neutral_short:
                desired_signal = -BASE_SIZE
            # Medium signal: HMA fast falling + daily bear
            elif hma_16_falling and rsi_bearish:
                desired_signal = -BASE_SIZE * 0.7
            # Weak signal: Just daily bear + HMA fast below slow (ensures trades)
            elif hma_fast_below_slow:
                desired_signal = -BASE_SIZE * 0.5
            # Fallback: Daily bear + weekly bear + RSI < 55 (very loose)
            elif weekly_bear and rsi[i] < 55.0:
                desired_signal = -BASE_SIZE * 0.4
        
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
        elif desired_signal >= BASE_SIZE * 0.6:
            final_signal = BASE_SIZE * 0.7
        elif desired_signal >= BASE_SIZE * 0.35:
            final_signal = BASE_SIZE * 0.5
        elif desired_signal <= -BASE_SIZE * 0.85:
            final_signal = -BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.6:
            final_signal = -BASE_SIZE * 0.7
        elif desired_signal <= -BASE_SIZE * 0.35:
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
2026-03-24 01:24
