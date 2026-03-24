# Strategy: mtf_12h_kama_crsi_chop_1d1w_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.452 | -9.7% | -20.6% | 164 | FAIL |
| ETHUSDT | -0.106 | +6.4% | -28.2% | 154 | FAIL |
| SOLUSDT | 0.411 | +64.5% | -36.9% | 180 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.129 | +6.7% | -19.1% | 55 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #472: 12h Primary + 1d/1w HTF — KAMA Adaptive Trend + Connors RSI + Choppiness Regime

Hypothesis: Based on research showing Connors RSI (CRSI) achieves 75% win rate for mean reversion,
combined with Choppiness Index for regime detection and KAMA for adaptive trend following.
Key innovations:
1. KAMA (Kaufman Adaptive MA) - adapts speed based on volatility (ER ratio)
2. Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3 - proven reversal signal
3. Choppiness Index (14) - regime filter: >61.8 = range (mean revert), <38.2 = trend
4. 1d KAMA + 1w KAMA for HTF bias alignment
5. ATR(14) trailing stop at 2.5x for risk management
6. Discrete position sizing: 0.0, ±0.25, ±0.30 to minimize fee churn

Why this should work: CRSI catches reversals better than standard RSI (proven in bear markets).
Choppiness Index prevents trend strategies in range markets (major source of losses).
KAMA adapts to volatility - faster in trends, slower in chop. 12h TF = 20-50 trades/year target.
Target: Sharpe > 0.612, DD < -35%, trades >= 30 on train, >= 3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_crsi_chop_1d1w_v1"
timeframe = "12h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average."""
    n = len(close)
    kama = np.full(n, np.nan)
    
    # Efficiency Ratio
    change = np.abs(close - np.roll(close, period))
    change[0:period] = np.nan
    volatility = np.zeros(n)
    for i in range(period, n):
        volatility[i] = np.sum(np.abs(np.diff(close[i-period:i+1])))
    
    with np.errstate(divide='ignore', invalid='ignore'):
        er = change / (volatility + 1e-10)
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    sc = np.square(er) * (fast_sc - slow_sc) + slow_sc
    
    # KAMA calculation
    kama[period] = close[period]
    for i in range(period + 1, n):
        if np.isnan(kama[i-1]):
            continue
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_rsi(close, period):
    """Calculate RSI using Wilder's smoothing."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_s = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_s = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = gain_s / (loss_s + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """Calculate Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3"""
    n = len(close)
    crsi = np.full(n, np.nan)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI of streak
    streak_abs = np.abs(streak)
    streak_rsi = calculate_rsi(streak_abs + 1e-10, streak_period)
    # Adjust for direction
    streak_rsi = np.where(streak >= 0, streak_rsi, 100 - streak_rsi)
    
    # Percent Rank (100)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window[:-1] < current) / (rank_period - 1) * 100
        crsi[i] = (rsi_short[i] + streak_rsi[i] + rank) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index (CHOP). High = choppy, Low = trending."""
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        atr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        atr_avg = atr_sum / period
        
        if highest - lowest > 1e-10 and atr_avg > 1e-10:
            chop[i] = 100.0 * np.log10((highest - lowest) / (atr_avg * np.sqrt(period))) / np.log10(period)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
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
    
    # Calculate 12h indicators (primary timeframe)
    kama_12h = calculate_kama(close, period=10)
    kama_50 = calculate_kama(close, period=50)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, 14)
    
    # Calculate and align HTF indicators
    kama_1d_raw = calculate_kama(df_1d['close'].values, period=21)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    kama_1w_raw = calculate_kama(df_1w['close'].values, period=21)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w_raw)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.30
    SIZE_SHORT = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(kama_12h[i]) or np.isnan(kama_50[i]):
            continue
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            continue
        if np.isnan(kama_1d_aligned[i]) or np.isnan(kama_1w_aligned[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 55.0  # Range market
        is_trending = chop[i] < 45.0  # Trend market
        
        # === PRIMARY TREND (KAMA crossover) ===
        trend_bullish = kama_12h[i] > kama_50[i]
        trend_bearish = kama_12h[i] < kama_50[i]
        
        # === HTF TREND BIAS (1d + 1w KAMA) ===
        price_above_kama_1d = close[i] > kama_1d_aligned[i]
        price_below_kama_1d = close[i] < kama_1d_aligned[i]
        price_above_kama_1w = close[i] > kama_1w_aligned[i]
        price_below_kama_1w = close[i] < kama_1w_aligned[i]
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi[i] < 15.0  # Strong reversal long
        crsi_overbought = crsi[i] > 85.0  # Strong reversal short
        crsi_neutral_long = 20.0 < crsi[i] < 50.0  # Trend continuation long
        crsi_neutral_short = 50.0 < crsi[i] < 80.0  # Trend continuation short
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRIES
        long_score = 0
        
        # In choppy market: mean reversion (CRSI oversold)
        if is_choppy and crsi_oversold:
            long_score += 3
        
        # In trending market: trend continuation
        if is_trending and trend_bullish:
            long_score += 2
            if crsi_neutral_long:
                long_score += 1
        
        # HTF alignment bonus
        if price_above_kama_1d:
            long_score += 1
        if price_above_kama_1w:
            long_score += 1
        
        if long_score >= 3:
            desired_signal = SIZE_LONG
        
        # SHORT ENTRIES
        if desired_signal == 0.0:
            short_score = 0
            
            # In choppy market: mean reversion (CRSI overbought)
            if is_choppy and crsi_overbought:
                short_score += 3
            
            # In trending market: trend continuation
            if is_trending and trend_bearish:
                short_score += 2
                if crsi_neutral_short:
                    short_score += 1
            
            # HTF alignment bonus
            if price_below_kama_1d:
                short_score += 1
            if price_below_kama_1w:
                short_score += 1
            
            if short_score >= 3:
                desired_signal = -SIZE_SHORT
        
        # === STOPLOSS CHECK (Trailing ATR) ===
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
        
        # === HOLD LOGIC — Maintain position if trend unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and trend_bullish and price_above_kama_1d:
                desired_signal = SIZE_LONG
            elif position_side < 0 and trend_bearish and price_below_kama_1d:
                desired_signal = -SIZE_SHORT
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = 0.30
        elif desired_signal < 0:
            desired_signal = -0.25
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals
```

## Last Updated
2026-03-23 11:00
