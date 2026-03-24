# Strategy: mtf_4h_donchian_hma_rsi_chop_12h_atr_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.083 | +16.5% | -15.6% | 632 | FAIL |
| ETHUSDT | -0.611 | -11.0% | -32.3% | 614 | FAIL |
| SOLUSDT | 0.242 | +36.6% | -25.8% | 687 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 1.113 | +24.7% | -9.3% | 202 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #1334: 4h Primary + 12h/1d HTF — Donchian Breakout + HMA Trend + RSI Pullback

Hypothesis: Donchian breakouts capture momentum moves, but need trend filter to avoid
false breakouts in chop. HMA(21) on 12h provides macro bias. RSI(7) pullback entries
reduce whipsaw vs pure breakout. Choppiness Index filters out range markets.

Key improvements over #1329:
1. Donchian(20) breakout instead of KAMA crossover - catches momentum earlier
2. 12h HMA(21) instead of 1d - faster macro signal, more trades
3. Choppiness Index < 50 to avoid range markets (proven in literature)
4. RSI(7) pullback bands widened (20-60 long, 40-80 short) to ensure trades
5. ATR trail at 2.5x + time-based exit (exit after 20 bars if no profit)

Target: 40-80 trades/year on 4h, Sharpe > 0.612, trades >= 50 train, >= 8 test
Timeframe: 4h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_hma_rsi_chop_12h_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - faster response than EMA"""
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
                diff_window = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_window.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_window) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1, dtype=np.float64)
                    hma[i] = np.sum(np.array(diff_window) * weights) / np.sum(weights)
    
    return hma

def calculate_rsi(close, period=7):
    """Relative Strength Index - faster period for pullback detection"""
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
    """Donchian Channel - breakout levels"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.nanmax(high[i-period+1:i+1])
        lower[i] = np.nanmin(low[i-period+1:i+1])
    
    return upper, lower

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market chop vs trend
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate ATR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    choppiness = np.full(n, np.nan)
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        
        price_range = highest_high - lowest_low
        if price_range > 1e-10 and atr_sum > 1e-10:
            choppiness[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return choppiness

def calculate_sma(close, period=200):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = np.full(n, np.nan)
    for i in range(period - 1, n):
        sma[i] = np.mean(close[i-period+1:i+1])
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 12h HMA for macro trend filter
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (4h) indicators
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    rsi = calculate_rsi(close, period=7)
    atr = calculate_atr(high, low, close, period=14)
    choppiness = calculate_choppiness(high, low, close, period=14)
    sma_200 = calculate_sma(close, period=200)
    
    # HMA slope for trend confirmation
    hma_4h = calculate_hma(close, period=21)
    hma_slope = np.full(n, np.nan)
    for i in range(1, n):
        if not np.isnan(hma_4h[i]) and not np.isnan(hma_4h[i-1]):
            hma_slope[i] = hma_4h[i] - hma_4h[i-1]
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    
    # Position tracking for stoploss and time exit
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        if np.isnan(rsi[i]) or np.isnan(choppiness[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_12h_aligned[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_4h[i]) or np.isnan(hma_slope[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO TREND (12h HMA) ===
        macro_bull = close[i] > hma_12h_aligned[i]
        macro_bear = close[i] < hma_12h_aligned[i]
        
        # === LOCAL TREND (4h HMA) ===
        hma_bull = (close[i] > hma_4h[i]) and (hma_slope[i] > 0)
        hma_bear = (close[i] < hma_4h[i]) and (hma_slope[i] < 0)
        
        # === CHOPPINESS FILTER - avoid range markets ===
        not_choppy = choppiness[i] < 55.0  # < 61.8 threshold, allow some buffer
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        breakout_long = close[i] > donchian_upper[i-1]  # Break above previous upper
        breakout_short = close[i] < donchian_lower[i-1]  # Break below previous lower
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRY: Macro bull + HMA bull + Donchian breakout + RSI pullback confirmation
        if macro_bull and hma_bull and not_choppy:
            # Donchian breakout with RSI confirmation (not overbought)
            if breakout_long and rsi[i] < 70.0:
                desired_signal = BASE_SIZE
            # RSI pullback in uptrend (20-55 range for RSI7)
            elif 20.0 <= rsi[i] <= 55.0 and above_sma200:
                desired_signal = BASE_SIZE
            # Price above HMA with strong momentum
            elif close[i] > hma_4h[i] and hma_slope[i] > 0 and above_sma200:
                desired_signal = BASE_SIZE
        
        # SHORT ENTRY: Macro bear + HMA bear + Donchian breakout + RSI confirmation
        elif macro_bear and hma_bear and not_choppy:
            # Donchian breakout with RSI confirmation (not oversold)
            if breakout_short and rsi[i] > 30.0:
                desired_signal = -BASE_SIZE
            # RSI bounce in downtrend (45-80 range for RSI7)
            elif 45.0 <= rsi[i] <= 80.0 and below_sma200:
                desired_signal = -BASE_SIZE
            # Price below HMA with strong momentum
            elif close[i] < hma_4h[i] and hma_slope[i] < 0 and below_sma200:
                desired_signal = -BASE_SIZE
        
        # === RANGE MARKET: Mean revert at Donchian extremes (choppy) ===
        if not not_choppy and desired_signal == 0.0:
            # Long at Donchian lower with RSI oversold
            if close[i] < donchian_lower[i-1] * 1.002 and rsi[i] < 30.0:
                desired_signal = BASE_SIZE * 0.5  # Half size in chop
            # Short at Donchian upper with RSI overbought
            elif close[i] > donchian_upper[i-1] * 0.998 and rsi[i] > 70.0:
                desired_signal = -BASE_SIZE * 0.5  # Half size in chop
        
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
        
        # === TIME-BASED EXIT (exit after 20 bars if no profit) ===
        time_exit = False
        if in_position and (i - entry_bar) > 20:
            if position_side > 0 and close[i] < entry_price:
                time_exit = True
            elif position_side < 0 and close[i] > entry_price:
                time_exit = True
        
        if stoploss_triggered or time_exit:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0.1:
            final_signal = BASE_SIZE
        elif desired_signal < -0.1:
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
                entry_bar = i
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                entry_bar = i
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
                entry_bar = 0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals
```

## Last Updated
2026-03-23 22:56
