# Strategy: mtf_12h_chop_regime_crsi_donchian_1d_hma_atr_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.571 | -4.9% | -12.6% | 386 | FAIL |
| ETHUSDT | 0.065 | +22.2% | -25.4% | 395 | PASS |
| SOLUSDT | 0.821 | +122.3% | -21.7% | 381 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | -1.143 | -11.7% | -14.7% | 131 | FAIL |
| SOLUSDT | 0.391 | +12.9% | -18.6% | 126 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #1226: 12h Primary + 1d HTF — Choppiness Index Regime + Connors RSI

Hypothesis: Research shows Choppiness Index regime switching achieved ETH Sharpe +0.923.
Connors RSI has 75% win rate for mean reversion. Combine these proven edges:
(1) CHOP(14) > 61.8 = range regime → mean revert with CRSI extremes
(2) CHOP(14) < 38.2 = trend regime → trend follow with Donchian breakout
(3) 1d HMA for macro trend filter (aligns with 12h direction)
(4) ATR trailing stop for risk management

Key differences from #1222:
- Connors RSI instead of regular RSI (faster, more sensitive to extremes)
- Choppiness Index for regime detection (proven edge on ETH)
- Dual logic: mean revert in chop, trend follow in trends
- Looser CRSI thresholds (15/85 instead of 10/90) to ensure trade frequency

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, DD > -50%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_chop_regime_crsi_donchian_1d_hma_atr_v1"
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

def calculate_rsi(close, period=3):
    """Relative Strength Index for Connors RSI component."""
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

def calculate_rsi_streak(close, period=2):
    """RSI of streak length (consecutive up/down days) for Connors RSI."""
    n = len(close)
    rsi_streak = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi_streak
    
    # Calculate streak lengths
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = max(1, streak[i-1] + 1) if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = min(-1, streak[i-1] - 1) if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    # Convert to positive values for RSI calculation
    streak_abs = np.abs(streak)
    
    # Calculate RSI on streak lengths
    delta = np.diff(streak_abs)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = loss_smooth > 1e-10
    rs = np.zeros(n)
    rs[mask] = gain_smooth[mask] / loss_smooth[mask]
    
    rsi_streak = 100.0 - (100.0 / (1.0 + rs))
    rsi_streak[:period] = np.nan
    
    return rsi_streak

def calculate_percent_rank(close, period=100):
    """Percentile rank for Connors RSI component."""
    n = len(close)
    pr = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
        current = close[i]
        rank = np.sum(window < current) / period * 100.0
        pr[i] = rank
    
    return pr

def calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3"""
    rsi = calculate_rsi(close, rsi_period)
    rsi_streak = calculate_rsi_streak(close, streak_period)
    pr = calculate_percent_rank(close, pr_period)
    
    crsi = (rsi + rsi_streak + pr) / 3.0
    return crsi

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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) — measures market choppy vs trending.
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = range/choppy, CHOP < 38.2 = trending
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    atr = calculate_atr(high, low, close, period)
    
    for i in range(period, n):
        if np.isnan(atr[i]):
            continue
        
        atr_sum = np.nansum(atr[i - period + 1:i + 1])
        highest = np.nanmax(high[i - period + 1:i + 1])
        lowest = np.nanmin(low[i - period + 1:i + 1])
        
        range_hl = highest - lowest
        if range_hl > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / range_hl) / np.log10(period)
    
    return chop

def calculate_donchian(high, low, period=20):
    """Donchian Channel — highest high and lowest low over period."""
    n = len(close) if 'close' in dir() else len(high)
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.nanmax(high[i - period + 1:i + 1])
        lower[i] = np.nanmin(low[i - period + 1:i + 1])
    
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
    hma_12h = calculate_hma(close, period=21)
    atr = calculate_atr(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # Slightly higher than 0.25 for better returns
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(crsi[i]) or atr[i] <= 1e-10:
            continue
        if np.isnan(hma_12h[i]) or np.isnan(chop[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(donchian_upper[i]):
            continue
        
        # === MACRO TREND (1d HMA) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 55.0  # Slightly lower threshold for more regime switches
        is_trending = chop[i] < 45.0  # Slightly higher threshold for more trend signals
        
        # === CONNORS RSI EXTREMES (looser for more trades) ===
        crsi_oversold = crsi[i] < 20.0  # Looser than 10 for more trades
        crsi_overbought = crsi[i] > 80.0  # Looser than 90 for more trades
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_up = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        donchian_breakout_down = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === HMA SLOPE (5-bar lookback) ===
        hma_slope = 0.0
        if not np.isnan(hma_12h[i]) and not np.isnan(hma_12h[i-5]):
            hma_slope = hma_12h[i] - hma_12h[i-5]
        
        trend_up = hma_slope > 0
        trend_down = hma_slope < 0
        
        # === ENTRY CONDITIONS (Dual Regime Logic) ===
        desired_signal = 0.0
        
        # LONG ENTRIES
        if is_choppy:
            # Mean reversion in choppy market
            if crsi_oversold and macro_bull:
                desired_signal = BASE_SIZE
            elif crsi_oversold and trend_up:
                desired_signal = BASE_SIZE
        elif is_trending:
            # Trend following in trending market
            if donchian_breakout_up and macro_bull:
                desired_signal = BASE_SIZE
            elif trend_up and macro_bull and crsi[i] < 50:
                # Pullback entry in uptrend
                desired_signal = BASE_SIZE
        else:
            # Neutral regime — use simpler conditions
            if crsi_oversold and macro_bull:
                desired_signal = BASE_SIZE
            elif trend_up and macro_bull:
                desired_signal = BASE_SIZE
        
        # SHORT ENTRIES
        if is_choppy:
            # Mean reversion in choppy market
            if crsi_overbought and macro_bear:
                desired_signal = -BASE_SIZE
            elif crsi_overbought and trend_down:
                desired_signal = -BASE_SIZE
        elif is_trending:
            # Trend following in trending market
            if donchian_breakout_down and macro_bear:
                desired_signal = -BASE_SIZE
            elif trend_down and macro_bear and crsi[i] > 50:
                # Pullback entry in downtrend
                desired_signal = -BASE_SIZE
        else:
            # Neutral regime — use simpler conditions
            if crsi_overbought and macro_bear:
                desired_signal = -BASE_SIZE
            elif trend_down and macro_bear:
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
2026-03-23 21:32
