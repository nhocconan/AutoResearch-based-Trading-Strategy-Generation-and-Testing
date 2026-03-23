# Strategy: mtf_1d_zscore_weekly_hma_rsi_atr_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.135 | +0.9% | -5.4% | 44 | FAIL |
| ETHUSDT | -0.613 | +6.0% | -10.1% | 48 | FAIL |
| SOLUSDT | 0.221 | +28.9% | -3.6% | 45 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.242 | +8.6% | -6.0% | 23 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #150: 1d Z-Score Mean Reversion with Weekly HMA Trend Filter
Hypothesis: Daily timeframe with Z-score mean reversion works better than pure trend
following in bear/range markets (2022, 2025). Z-score(20) identifies extreme deviations
from recent mean. Weekly HMA provides major trend bias - only take long mean-reversion
when weekly trend is bullish, short when bearish. This avoids counter-trend trades that
get stopped out. Simpler entry conditions (Z<-2 or Z>2 + RSI confirmation) ensure
sufficient trades while maintaining quality. ATR stoploss at 2.5*ATR protects capital.
Position sizing: 0.28 entry, 0.14 at 2R profit, discrete levels minimize fee churn.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_zscore_weekly_hma_rsi_atr_v1"
timeframe = "1d"
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
    half = period // 2
    if half < 1:
        half = 1
    sqrt_period = int(np.sqrt(period))
    if sqrt_period < 1:
        sqrt_period = 1
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

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

def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion signals."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    zscore = np.where(std > 0, (close - sma) / std, 0.0)
    return zscore

def calculate_momentum(close, period=10):
    """Calculate Rate of Change momentum."""
    momentum = np.zeros(len(close))
    for i in range(period, len(close)):
        if close[i-period] > 0:
            momentum[i] = (close[i] - close[i-period]) / close[i-period] * 100
        else:
            momentum[i] = 0.0
    return momentum

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    zscore = calculate_zscore(close, 20)
    momentum = calculate_momentum(close, 10)
    hma_20 = calculate_hma(close, 20)
    hma_50 = calculate_hma(close, 50)
    
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
        # Weekly trend filter (major trend direction)
        weekly_bullish = hma_1w_aligned[i] > 0 and close[i] > hma_1w_aligned[i]
        weekly_bearish = hma_1w_aligned[i] > 0 and close[i] < hma_1w_aligned[i]
        
        # Daily trend filter
        daily_bullish = hma_20[i] > hma_50[i]
        daily_bearish = hma_20[i] < hma_50[i]
        
        # Z-score mean reversion signals
        zscore_extreme_low = zscore[i] < -1.8
        zscore_extreme_high = zscore[i] > 1.8
        zscore_neutral = -1.0 < zscore[i] < 1.0
        
        # RSI confirmation (wider thresholds for more trades)
        rsi_oversold = rsi[i] < 40
        rsi_overbought = rsi[i] > 60
        rsi_rising = rsi[i] > rsi[i-3] if i > 3 else False
        rsi_falling = rsi[i] < rsi[i-3] if i > 3 else False
        
        # Momentum confirmation
        mom_positive = momentum[i] > -5.0
        mom_negative = momentum[i] < 5.0
        
        new_signal = 0.0
        
        # LONG ENTRY: Z-score extreme low + RSI oversold + Weekly not bearish
        if zscore_extreme_low and rsi_oversold:
            if weekly_bullish or (not weekly_bearish and daily_bullish):
                new_signal = SIZE_ENTRY
            elif not weekly_bearish and mom_positive:
                new_signal = SIZE_ENTRY
        
        # SHORT ENTRY: Z-score extreme high + RSI overbought + Weekly not bullish
        elif zscore_extreme_high and rsi_overbought:
            if weekly_bearish or (not weekly_bullish and daily_bearish):
                new_signal = -SIZE_ENTRY
            elif not weekly_bullish and mom_negative:
                new_signal = -SIZE_ENTRY
        
        # TREND FOLLOWING: HMA crossover with momentum
        elif daily_bullish and hma_20[i-1] <= hma_50[i-1] and rsi_rising:
            if weekly_bullish or zscore_neutral:
                new_signal = SIZE_ENTRY
        
        elif daily_bearish and hma_20[i-1] >= hma_50[i-1] and rsi_falling:
            if weekly_bearish or zscore_neutral:
                new_signal = -SIZE_ENTRY
        
        # Stoploss logic (Rule 6) - check BEFORE updating position tracking
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
2026-03-22 02:43
