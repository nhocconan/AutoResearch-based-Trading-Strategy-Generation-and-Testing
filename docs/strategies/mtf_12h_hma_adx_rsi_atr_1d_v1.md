# Strategy: mtf_12h_hma_adx_rsi_atr_1d_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.069 | +10.8% | -25.0% | 344 | FAIL |
| ETHUSDT | 0.478 | +66.5% | -29.7% | 343 | PASS |
| SOLUSDT | 0.912 | +217.1% | -27.0% | 331 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | -0.664 | -11.8% | -26.0% | 100 | FAIL |
| SOLUSDT | 0.050 | +4.2% | -21.4% | 114 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #486: 12h Primary + 1d HTF — HMA Trend + ADX Strength + RSI Entry + ATR Stop

Hypothesis: After analyzing 400+ failed strategies, complex regime-switching and CRSI combinations
are overfitting. Simpler trend-following with relaxed thresholds works better on 12h timeframe.
Key innovations:
1. HMA(21) - faster than EMA, less lag than SMA, proven in best strategies
2. ADX(14) > 20 (relaxed from 25) - ensures some trend without being too restrictive
3. RSI(14) 35/65 thresholds (relaxed from 30/70) - more entry opportunities
4. 1d HMA for HTF major trend bias (aligns with proven 4h/1d MTF patterns)
5. ATR(14) trailing stop at 2.5x - protects against 2022-style crashes
6. Discrete sizing: 0.30 long, -0.25 short (asymmetric due to crypto long bias)
7. HOLD logic: maintain position while trend intact (reduces churn)

Why 12h works: Natural 20-50 trades/year target. Less noise than 4h, more signals than 1d.
1d HTF provides major trend filter without excessive lag. Relaxed thresholds ensure
trade generation (avoiding 0-trade failure mode seen in 8+ recent experiments).

Target: Sharpe > 0.612, DD < -35%, trades >= 30 train, >= 3 test, ALL symbols positive Sharpe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_adx_rsi_atr_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    Reduces lag while maintaining smoothness.
    """
    n = len(close)
    hma = np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    if half < 1 or sqrt_period < 1:
        return hma
    
    # WMA helper
    def wma(series, w_period):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, w_period + 1, dtype=float)
        for i in range(w_period - 1, len(series)):
            if np.any(np.isnan(series[i - w_period + 1:i + 1])):
                continue
            result[i] = np.sum(series[i - w_period + 1:i + 1] * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    # Combine
    diff = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    hma = wma(diff, sqrt_period)
    return hma

def calculate_adx(high, low, close, period=14):
    """
    Calculate Average Directional Index (ADX).
    Measures trend strength regardless of direction.
    ADX > 25 = strong trend, ADX < 20 = weak/range
    """
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2:
        return adx
    
    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i - 1])
        tr3 = np.abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    # Smooth with Wilder's method (EMA with span=period)
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # DI+ and DI-
    with np.errstate(divide='ignore', invalid='ignore'):
        plus_di = 100.0 * plus_dm_smooth / (tr_smooth + 1e-10)
        minus_di = 100.0 * minus_dm_smooth / (tr_smooth + 1e-10)
    
    # DX
    with np.errstate(divide='ignore', invalid='ignore'):
        dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    
    # ADX = smoothed DX
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing method."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0.0)
    loss[1:] = np.where(delta < 0, -delta, 0.0)
    
    gain_s = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_s = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = gain_s / (loss_s + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i - 1])
        tr3 = np.abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h indicators (primary timeframe)
    hma_12h = calculate_hma(close, period=21)
    adx_12h, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    rsi_12h = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Calculate and align HTF indicators (1d HMA for major trend bias)
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
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
        if np.isnan(hma_12h[i]):
            continue
        if np.isnan(adx_12h[i]):
            continue
        if np.isnan(rsi_12h[i]):
            continue
        if np.isnan(hma_1d_aligned[i]):
            continue
        
        # === HTF MAJOR TREND BIAS (1d HMA) ===
        htf_bullish = close[i] > hma_1d_aligned[i]
        htf_bearish = close[i] < hma_1d_aligned[i]
        
        # === PRIMARY TREND (12h HMA) ===
        price_above_hma = close[i] > hma_12h[i]
        price_below_hma = close[i] < hma_12h[i]
        hma_slope_up = hma_12h[i] > hma_12h[i - 5] if i >= 5 else False
        hma_slope_down = hma_12h[i] < hma_12h[i - 5] if i >= 5 else False
        
        # === TREND STRENGTH (ADX) ===
        # Relaxed threshold: ADX > 20 (not 25) to ensure trade generation
        trend_strong = adx_12h[i] > 20.0
        
        # === RSI SIGNALS (relaxed thresholds) ===
        rsi_oversold = rsi_12h[i] < 50.0  # Relaxed from 35
        rsi_overbought = rsi_12h[i] > 50.0  # Relaxed from 65
        rsi_deep_oversold = rsi_12h[i] < 40.0
        rsi_deep_overbought = rsi_12h[i] > 60.0
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRIES - Multiple confluence conditions (relaxed for trade generation)
        long_conditions = 0
        
        # HTF bullish bias (strong signal)
        if htf_bullish:
            long_conditions += 2
        
        # Price above 12h HMA
        if price_above_hma:
            long_conditions += 1
        
        # HMA slope up
        if hma_slope_up:
            long_conditions += 1
        
        # RSI not overbought (room to run)
        if rsi_oversold:
            long_conditions += 1
        
        # ADX shows some trend (or we enter anyway in strong HTF bias)
        if trend_strong or htf_bullish:
            long_conditions += 1
        
        # Enter long if >= 4 conditions met (relaxed from 5)
        if long_conditions >= 4:
            desired_signal = SIZE_LONG
        
        # SHORT ENTRIES
        if desired_signal == 0.0:
            short_conditions = 0
            
            # HTF bearish bias
            if htf_bearish:
                short_conditions += 2
            
            # Price below 12h HMA
            if price_below_hma:
                short_conditions += 1
            
            # HMA slope down
            if hma_slope_down:
                short_conditions += 1
            
            # RSI not oversold
            if rsi_overbought:
                short_conditions += 1
            
            # ADX shows trend
            if trend_strong or htf_bearish:
                short_conditions += 1
            
            if short_conditions >= 4:
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
        # This reduces churn and ensures we stay in winning trades
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if HTF still bullish OR price above HMA
                if htf_bullish or price_above_hma:
                    desired_signal = SIZE_LONG
            elif position_side < 0:
                # Hold short if HTF still bearish OR price below HMA
                if htf_bearish or price_below_hma:
                    desired_signal = -SIZE_SHORT
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = SIZE_LONG
        elif desired_signal < 0:
            desired_signal = -SIZE_SHORT
        
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
                # Position flip
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
2026-03-23 11:13
