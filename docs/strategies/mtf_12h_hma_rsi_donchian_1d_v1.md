# Strategy: mtf_12h_hma_rsi_donchian_1d_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.931 | -33.8% | -36.2% | 235 | FAIL |
| ETHUSDT | -0.181 | -1.7% | -19.1% | 208 | FAIL |
| SOLUSDT | 0.631 | +118.4% | -32.2% | 212 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.467 | +16.7% | -12.9% | 54 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #466: 12h Primary + 1d HTF — HMA Trend + RSI Momentum + Donchian Breakout

Hypothesis: Based on proven patterns showing 12h timeframe works best for swing trading.
Simpler is better — many failed strategies had too many filters causing 0 trades.
Key innovations:
1. HMA(21)/HMA(50) crossover for clean trend detection (faster than EMA)
2. RSI(14) momentum filter — 45-65 range for trend continuation (not extremes)
3. Donchian(20) breakout for momentum confirmation
4. 1d HMA(21) for higher timeframe bias alignment
5. ATR(14) trailing stop at 2.5x for risk management
6. Discrete position sizing: 0.0, ±0.25, ±0.30 to minimize fee churn

Target: Sharpe > 0.612, 30-60 trades/year, DD < -35%
Timeframe: 12h (proven best for crypto swing trading)
Why this should work: Fewer filters = more trades. HMA is faster than EMA for crypto.
RSI mid-range (not extremes) captures trend continuation better than reversal.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_rsi_donchian_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = period // 2
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    diff = 2.0 * wma1 - wma2
    sqrt_period = int(np.sqrt(period))
    hma = diff.ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

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

def calculate_donchian(high, low, period):
    """Calculate Donchian Channel (upper and lower bounds)."""
    n = len(high)
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
    
    # Calculate 12h indicators (primary timeframe)
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    rsi_14 = calculate_rsi(close, 14)
    atr_14 = calculate_atr(high, low, close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(hma_21[i]) or np.isnan(hma_50[i]):
            continue
        if np.isnan(rsi_14[i]) or np.isnan(donchian_upper[i]):
            continue
        if np.isnan(hma_1d_aligned[i]):
            continue
        
        # === PRIMARY TREND (HMA crossover) ===
        trend_bullish = hma_21[i] > hma_50[i]
        trend_bearish = hma_21[i] < hma_50[i]
        
        # === HTF TREND BIAS (1d HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === MOMENTUM (RSI mid-range for trend continuation) ===
        rsi_momentum_long = 45.0 < rsi_14[i] < 70.0
        rsi_momentum_short = 30.0 < rsi_14[i] < 55.0
        
        # === BREAKOUT (Donchian) ===
        breakout_long = close[i] >= donchian_upper[i] * 0.998  # Near upper bound
        breakout_short = close[i] <= donchian_lower[i] * 1.002  # Near lower bound
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG: Trend bullish + HTF bullish + RSI momentum + breakout
        long_score = 0
        if trend_bullish:
            long_score += 2
        if price_above_hma_1d:
            long_score += 1
        if rsi_momentum_long:
            long_score += 1
        if breakout_long:
            long_score += 1
        
        if long_score >= 3:
            desired_signal = SIZE_LONG
        
        # SHORT: Trend bearish + HTF bearish + RSI momentum + breakout
        if desired_signal == 0.0:
            short_score = 0
            if trend_bearish:
                short_score += 2
            if price_below_hma_1d:
                short_score += 1
            if rsi_momentum_short:
                short_score += 1
            if breakout_short:
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
            if position_side > 0 and trend_bullish and price_above_hma_1d:
                desired_signal = SIZE_LONG
            elif position_side < 0 and trend_bearish and price_below_hma_1d:
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
2026-03-23 10:54
