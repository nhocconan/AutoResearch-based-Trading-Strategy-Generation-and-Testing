# Strategy: mtf_12h_donchian_hma_1d_rsi_atr_clean_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.071 | +23.2% | -16.9% | 183 | PASS |
| ETHUSDT | 0.195 | +30.4% | -19.7% | 198 | PASS |
| SOLUSDT | 1.310 | +208.1% | -20.1% | 169 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.069 | -3.6% | -9.4% | 64 | FAIL |
| ETHUSDT | -1.056 | -10.0% | -17.8% | 61 | FAIL |
| SOLUSDT | 0.074 | +6.3% | -12.3% | 59 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #1396: 12h Primary + 1d HTF — Clean Donchian Breakout with HMA Trend Filter

Hypothesis: Previous 12h failure (#1392 Sharpe=0.305) was caused by over-complicated regime
filters (Choppiness + CRSI) that over-filtered signals. The working 4h pattern (#1391) and
current best 1d (Sharpe=0.618) both used clean trend following: Donchian + HMA + RSI + ATR.

Key insight: 12h timeframe needs SIMPLE entry conditions to maintain trade frequency.
Adding Choppiness/CRSI regime filters reduced trades below minimum threshold.
Using proven 1d HMA(21) for macro bias + 12h Donchian(20/55) breakout entries.

Design:
1. 1d HMA(21) = macro trend direction (primary filter, proven on 4h/1d)
2. 12h Donchian(20/55) breakout = entry triggers (dual period for flexibility)
3. RSI(14) momentum confirmation (WIDE bands 30-70 to ensure >=30 trades/train)
4. ATR(14) trailing stop 2.5x = risk management
5. Position size 0.25 = conservative for 12h volatility
6. NO regime filter (Choppiness/CRSI failed on 12h in #1392)
7. Multiple entry paths per direction = ensures trade frequency

Target: 20-50 trades/year, Sharpe > 0.618 (beat 1d baseline), trades >= 30 train, >= 5 test
Timeframe: 12h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_hma_1d_rsi_atr_clean_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - faster response than EMA, less lag"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        weights = np.arange(1, span + 1, dtype=np.float64)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            if not np.any(np.isnan(window)):
                result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    hma = np.full(n, np.nan)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            if i >= sqrt_period - 1:
                diff_vals = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_vals.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_vals) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1, dtype=np.float64)
                    hma[i] = np.sum(np.array(diff_vals) * weights) / np.sum(weights)
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index - wide bands for entry confirmation"""
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
    """Average True Range - for stoploss sizing"""
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
    """Donchian Channel - breakout levels for entry trigger"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.nanmax(high[i-period+1:i+1])
        lower[i] = np.nanmin(low[i-period+1:i+1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for macro trend filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (12h) indicators
    donchian_20_upper, donchian_20_lower = calculate_donchian(high, low, period=20)
    donchian_55_upper, donchian_55_lower = calculate_donchian(high, low, period=55)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    
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
        if np.isnan(donchian_20_upper[i]) or np.isnan(donchian_55_upper[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]):
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
        
        # === MACRO TREND (1d HMA) - primary filter ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === RSI MOMENTUM (WIDE bands to ensure trades) ===
        rsi_bull = rsi[i] > 30.0
        rsi_bear = rsi[i] < 70.0
        rsi_strong_bull = rsi[i] > 45.0
        rsi_strong_bear = rsi[i] < 55.0
        rsi_neutral_bull = rsi[i] > 40.0
        rsi_neutral_bear = rsi[i] < 60.0
        
        # === DUAL DONCHIAN BREAKOUT ===
        breakout_20_long = close[i] > donchian_20_upper[i-1]
        breakout_20_short = close[i] < donchian_20_lower[i-1]
        breakout_55_long = close[i] > donchian_55_upper[i-1]
        breakout_55_short = close[i] < donchian_55_lower[i-1]
        
        # === DESIRED SIGNAL - MULTIPLE ENTRY PATHS PER DIRECTION ===
        desired_signal = 0.0
        
        # LONG ENTRY PATHS (any one triggers entry)
        # Path 1: Donchian-20 breakout + 1d trend + RSI (quick entry)
        if breakout_20_long and macro_bull and rsi_bull:
            desired_signal = BASE_SIZE
        # Path 2: Donchian-55 breakout + RSI strong (strong breakout)
        elif breakout_55_long and rsi_strong_bull:
            desired_signal = BASE_SIZE
        # Path 3: Price above 1d HMA + RSI momentum (trend continuation)
        elif macro_bull and rsi_neutral_bull:
            desired_signal = BASE_SIZE * 0.5
        # Path 4: Donchian-20 breakout alone (breakout momentum)
        elif breakout_20_long and rsi[i] > 35.0:
            desired_signal = BASE_SIZE * 0.5
        
        # SHORT ENTRY PATHS (any one triggers entry)
        # Path 1: Donchian-20 breakout + 1d trend + RSI (quick entry)
        elif breakout_20_short and macro_bear and rsi_bear:
            desired_signal = -BASE_SIZE
        # Path 2: Donchian-55 breakout + RSI strong (strong breakout)
        elif breakout_55_short and rsi_strong_bear:
            desired_signal = -BASE_SIZE
        # Path 3: Price below 1d HMA + RSI momentum (trend continuation)
        elif macro_bear and rsi_neutral_bear:
            desired_signal = -BASE_SIZE * 0.5
        # Path 4: Donchian-20 breakout alone (breakout momentum)
        elif breakout_20_short and rsi[i] < 65.0:
            desired_signal = -BASE_SIZE * 0.5
        
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
        if desired_signal >= BASE_SIZE * 0.4:
            final_signal = BASE_SIZE if desired_signal > 0 else -BASE_SIZE
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
2026-03-23 23:44
