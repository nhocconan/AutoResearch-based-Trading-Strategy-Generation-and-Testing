# Strategy: mtf_12h_kama_adaptive_daily_weekly_hma_rsi_atr_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.636 | -18.7% | -30.5% | 323 | FAIL |
| ETHUSDT | -0.306 | -10.0% | -24.9% | 311 | FAIL |
| SOLUSDT | 0.574 | +101.0% | -24.3% | 301 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.034 | +4.4% | -18.2% | 102 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #275: 12h KAMA Adaptive Trend with Daily/Weekly HMA Filter
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to volatility better than fixed EMAs,
reducing whipsaws in 2022 crash and 2025 bear market. Daily HMA provides primary trend bias,
Weekly HMA confirms macro direction. RSI pullback entries in trending markets only.
Simple logic with fewer filters to ensure sufficient trades (>10 train, >3 test).
Position sizing: 0.28 entry, 0.14 half at 2R profit. Stoploss: 2.5*ATR trailing.
Target: Beat Sharpe=0.499 from current best (mtf_12h_supertrend_daily_hma_rsi_pullback_v2)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_adaptive_daily_weekly_hma_rsi_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_kama(close, period=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average - adapts to market noise."""
    close_s = pd.Series(close)
    change = np.abs(close_s.diff(period).values)
    volatility = pd.Series(np.abs(close_s.diff().values)).rolling(window=period, min_periods=period).sum().values
    er = np.where(volatility > 0, change / volatility, 0.0)
    sc = (er * (2.0 / (fast + 1) - 2.0 / (slow + 1)) + 2.0 / (slow + 1)) ** 2
    kama = np.zeros(len(close))
    kama[period] = close[period]
    for i in range(period + 1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def calculate_rsi(close, period=14):
    """Calculate RSI indicator."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_kama_slope(kama, lookback=3):
    """Calculate KAMA slope direction."""
    slope = np.zeros(len(kama))
    for i in range(lookback, len(kama)):
        if kama[i] > kama[i-lookback]:
            slope[i] = 1.0
        elif kama[i] < kama[i-lookback]:
            slope[i] = -1.0
        else:
            slope[i] = 0.0
    return slope

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    kama = calculate_kama(close, 10, 2, 30)
    kama_slope = calculate_kama_slope(kama, 3)
    
    # Track previous values
    prev_kama = np.roll(kama, 1)
    prev_kama[0] = kama[0]
    prev_rsi = np.roll(rsi, 1)
    prev_rsi[0] = rsi[0]
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.28
    SIZE_HALF = 0.14
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # HTF trend filters (simple - daily primary, weekly confirmation bonus)
        daily_bullish = close[i] > hma_1d_aligned[i]
        daily_bearish = close[i] < hma_1d_aligned[i]
        weekly_bullish = close[i] > hma_1w_aligned[i]
        weekly_bearish = close[i] < hma_1w_aligned[i]
        
        # KAMA trend signals
        kama_bullish = kama_slope[i] > 0 and close[i] > kama[i]
        kama_bearish = kama_slope[i] < 0 and close[i] < kama[i]
        kama_cross_up = prev_kama[i] >= close[i] and kama[i] < close[i]
        kama_cross_down = prev_kama[i] <= close[i] and kama[i] > close[i]
        
        # RSI pullback signals (looser thresholds for more trades)
        rsi_pullback_long = 35 < rsi[i] < 55 and prev_rsi[i] <= 35
        rsi_pullback_short = 45 < rsi[i] < 65 and prev_rsi[i] >= 65
        rsi_oversold = rsi[i] < 40
        rsi_overbought = rsi[i] > 60
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # KAMA trend + Daily HMA bullish + RSI pullback
        if kama_bullish and daily_bullish:
            if rsi_pullback_long or rsi_oversold:
                new_signal = SIZE_ENTRY
            elif close[i] > kama[i] and rsi[i] > 45:
                new_signal = SIZE_ENTRY
        
        # KAMA cross up with trend confirmation
        elif kama_cross_up:
            if daily_bullish and rsi[i] > 40:
                new_signal = SIZE_ENTRY
            elif weekly_bullish and rsi[i] > 45:
                new_signal = SIZE_ENTRY
        
        # Strong trend continuation (weekly confirmation)
        elif daily_bullish and weekly_bullish:
            if kama_slope[i] > 0 and rsi[i] > 50:
                new_signal = SIZE_ENTRY
        
        # === SHORT ENTRY ===
        # KAMA trend + Daily HMA bearish + RSI pullback
        if kama_bearish and daily_bearish:
            if rsi_pullback_short or rsi_overbought:
                new_signal = -SIZE_ENTRY
            elif close[i] < kama[i] and rsi[i] < 55:
                new_signal = -SIZE_ENTRY
        
        # KAMA cross down with trend confirmation
        elif kama_cross_down:
            if daily_bearish and rsi[i] < 60:
                new_signal = -SIZE_ENTRY
            elif weekly_bearish and rsi[i] < 55:
                new_signal = -SIZE_ENTRY
        
        # Strong trend continuation (weekly confirmation)
        elif daily_bearish and weekly_bearish:
            if kama_slope[i] < 0 and rsi[i] < 50:
                new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from highest)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from lowest)
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = entry_price - close[i]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reduced (take profit)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals
```

## Last Updated
2026-03-22 04:14
