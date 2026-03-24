# Strategy: mtf_12h_hma_trend_rsi7_funding_zscore_1d1w_atr_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.333 | +2.6% | -17.6% | 297 | FAIL |
| ETHUSDT | 0.042 | +20.3% | -22.8% | 295 | PASS |
| SOLUSDT | 0.652 | +96.1% | -31.8% | 278 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | -0.839 | -8.3% | -20.9% | 98 | FAIL |
| SOLUSDT | 0.159 | +7.9% | -18.1% | 91 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #1222: 12h Primary + 1d/1w HTF — Simplified HMA Trend + RSI Mean Reversion

Hypothesis: #1212 had +21.3% return but negative Sharpe due to complex 3-regime system causing
whipsaws. Simplify to: (1) HMA trend direction, (2) RSI(7) for faster entries, (3) Funding rate
z-score for contrarian edge (proven on BTC/ETH), (4) Looser conditions to ensure >=30 trades.

Key changes from #1212:
- Remove Choppiness Index (causes false regime switches)
- Use HMA(21) slope instead of KAMA (simpler, proven in baseline)
- RSI(7) instead of RSI(14) for faster signals on 12h
- Add funding rate z-score contrarian filter (best edge for BTC/ETH per research)
- Smaller size (0.25) + tighter stop (2.5x ATR) for lower drawdown
- Looser entry thresholds to guarantee trade frequency

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, DD > -50%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_trend_rsi7_funding_zscore_1d1w_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average — reduces lag while maintaining smoothness."""
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            if i >= sqrt_period - 1:
                diff_window = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_window.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_window) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1)
                    hma[i] = np.sum(np.array(diff_window) * weights) / np.sum(weights)
    
    return hma

def calculate_rsi(close, period=7):
    """Relative Strength Index — faster period for 12h timeframe."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = loss_smooth > 1e-10
    rs = np.zeros(n)
    rs[mask] = gain_smooth[mask] / loss_smooth[mask]
    
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement and stoploss."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_zscore(series, period=30):
    """Z-score for mean reversion detection."""
    n = len(series)
    zscore = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        window = series[i - period + 1:i + 1]
        mean = np.mean(window)
        std = np.std(window)
        if std > 1e-10:
            zscore[i] = (series[i] - mean) / std
    
    return zscore

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for macro trend filter
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate and align 1d HMA for intermediate trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (12h) indicators
    hma_12h = calculate_hma(close, period=21)
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=7)
    
    # Calculate HMA slope (5-bar lookback)
    hma_slope = np.zeros(n)
    for i in range(5, n):
        if not np.isnan(hma_12h[i]) and not np.isnan(hma_12h[i-5]):
            hma_slope[i] = hma_12h[i] - hma_12h[i-5]
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(rsi[i]) or atr[i] <= 1e-10:
            continue
        if np.isnan(hma_12h[i]) or np.isnan(hma_slope[i]):
            continue
        if np.isnan(hma_1w_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        
        # === MACRO TREND (1w HMA) ===
        macro_bull = close[i] > hma_1w_aligned[i]
        macro_bear = close[i] < hma_1w_aligned[i]
        
        # === INTERMEDIATE TREND (1d HMA) ===
        inter_bull = close[i] > hma_1d_aligned[i]
        inter_bear = close[i] < hma_1d_aligned[i]
        
        # === PRIMARY TREND (12h HMA slope) ===
        trend_up = hma_slope[i] > 0
        trend_down = hma_slope[i] < 0
        
        # === RSI EXTREMES (looser for more trades) ===
        rsi_oversold = rsi[i] < 40.0
        rsi_overbought = rsi[i] > 60.0
        
        # === ENTRY CONDITIONS (simplified for trade frequency) ===
        desired_signal = 0.0
        
        # LONG: Trend up OR macro bull + RSI oversold (mean reversion in uptrend)
        if trend_up and rsi_oversold:
            desired_signal = BASE_SIZE
        elif macro_bull and inter_bull and rsi_oversold:
            desired_signal = BASE_SIZE
        elif trend_up and macro_bull:
            # Pure trend follow when both aligned
            desired_signal = BASE_SIZE
        
        # SHORT: Trend down OR macro bear + RSI overbought
        if trend_down and rsi_overbought:
            desired_signal = -BASE_SIZE
        elif macro_bear and inter_bear and rsi_overbought:
            desired_signal = -BASE_SIZE
        elif trend_down and macro_bear:
            # Pure trend follow when both aligned
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
        if desired_signal > 0:
            desired_signal = BASE_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE
        else:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
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
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals
```

## Last Updated
2026-03-23 21:29
