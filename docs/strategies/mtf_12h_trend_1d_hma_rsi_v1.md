# Strategy: mtf_12h_trend_1d_hma_rsi_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.669 | +3.1% | -12.9% | 362 | FAIL |
| ETHUSDT | -1.102 | -14.0% | -22.0% | 381 | FAIL |
| SOLUSDT | 0.600 | +55.5% | -12.3% | 331 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.141 | +7.4% | -6.7% | 112 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #005: 12h Multi-Timeframe Trend-Follow with 1d HMA Bias
Hypothesis: 12h timeframe captures intermediate trends while 1d HMA provides regime filter.
Key insight: Previous failures used overly complex regime detection (Choppiness) or too-strict conditions (Vol Breakout = 0 trades).
This strategy uses simpler logic: 1d HMA for trend bias, 12h EMA pullback entries, RSI confirmation, ATR stops.
Position sizing: 0.25-0.30 discrete levels to minimize fee churn while controlling drawdown.
Timeframe: 12h (REQUIRED for exp#005), HTF: 1d via mtf_data helper.
Why this might work: 12h has fewer whipsaws than 1h/4h, 1d HMA smoother than 4h for regime detection.
Must generate 10+ trades on train, 3+ on test - entry conditions loosened vs failed experiments.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_trend_1d_hma_rsi_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    n = len(close)
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gains).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(losses).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    rsi[~mask] = 100.0
    
    return rsi

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_sma(close, period):
    """Calculate SMA."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion filter."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    zscore = (close - sma) / (std + 1e-10)
    return zscore

def calculate_supertrend(high, low, close, period=10, mult=3.0):
    """Calculate Supertrend for trend direction."""
    atr = calculate_atr(high, low, close, period)
    hl2 = (high + low) / 2
    
    upper = hl2 + mult * atr
    lower = hl2 - mult * atr
    
    supertrend = np.zeros(len(close))
    trend = np.ones(len(close))  # 1 = bullish, -1 = bearish
    
    supertrend[0] = upper[0]
    for i in range(1, len(close)):
        if trend[i-1] == 1:
            if close[i] < lower[i]:
                trend[i] = -1
                supertrend[i] = upper[i]
            else:
                trend[i] = 1
                supertrend[i] = max(lower[i], supertrend[i-1])
        else:
            if close[i] > upper[i]:
                trend[i] = 1
                supertrend[i] = lower[i]
            else:
                trend[i] = -1
                supertrend[i] = min(upper[i], supertrend[i-1])
    
    return supertrend, trend

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    ema_200 = calculate_ema(close, 200)
    sma_200 = calculate_sma(close, 200)
    zscore = calculate_zscore(close, 20)
    
    # Supertrend for trend confirmation
    supertrend, st_trend = calculate_supertrend(high, low, close, 10, 3.0)
    
    # HMA on 12h for faster trend
    hma_12h = calculate_hma(close, 21)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_HALF = 0.125
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(ema_21[i]):
            signals[i] = 0.0
            continue
        
        # 1d trend bias (HTF) - main regime filter
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # 12h trend confirmation
        bull_trend_12h = close[i] > ema_50[i] and ema_21[i] > ema_50[i]
        bear_trend_12h = close[i] < ema_50[i] and ema_21[i] < ema_50[i]
        
        # Supertrend confirmation
        st_bullish = st_trend[i] == 1
        st_bearish = st_trend[i] == -1
        
        # Long-term trend filter
        above_200 = not np.isnan(sma_200[i]) and close[i] > sma_200[i]
        below_200 = not np.isnan(sma_200[i]) and close[i] < sma_200[i]
        
        # RSI conditions - LOOSENED for more trades
        rsi_pullback_long = 30 < rsi[i] < 60  # Wider range for more entries
        rsi_bounce_short = 40 < rsi[i] < 70
        rsi_oversold = rsi[i] < 40
        rsi_overbought = rsi[i] > 60
        
        # Z-score filter - avoid extreme entries
        zscore_neutral = abs(zscore[i]) < 2.0
        
        # HMA crossover on 12h
        hma_cross_long = False
        hma_cross_short = False
        if i >= 1 and not np.isnan(hma_12h[i]) and not np.isnan(hma_12h[i-1]):
            hma_cross_long = hma_12h[i] > ema_50[i] and hma_12h[i-1] <= ema_50[i-1]
            hma_cross_short = hma_12h[i] < ema_50[i] and hma_12h[i-1] >= ema_50[i-1]
        
        # Price pullback to EMA21
        price_near_ema21_long = close[i] <= ema_21[i] * 1.02 and close[i] >= ema_21[i] * 0.98
        price_near_ema21_short = close[i] >= ema_21[i] * 0.98 and close[i] <= ema_21[i] * 1.02
        
        # Price action: higher low for long, lower high for short
        higher_low = False
        lower_high = False
        if i >= 3:
            higher_low = low[i] > low[i-3]
            lower_high = high[i] < high[i-3]
        
        new_signal = 0.0
        
        # === LONG ENTRIES (only when 1d bullish) ===
        if bull_trend_1d:
            # Primary: Pullback to EMA21 with RSI confirmation
            if price_near_ema21_long and rsi_pullback_long and above_200:
                new_signal = SIZE_BASE
            
            # Secondary: HMA crossover with 1d confirmation
            elif hma_cross_long and bull_trend_1d and st_bullish:
                new_signal = SIZE_BASE
            
            # Tertiary: RSI oversold bounce in uptrend
            elif rsi_oversold and bull_trend_12h and zscore_neutral:
                new_signal = SIZE_HALF
            
            # Momentum: Higher low with trend
            elif higher_low and bull_trend_12h and rsi[i] > 45:
                new_signal = SIZE_HALF
        
        # === SHORT ENTRIES (only when 1d bearish) ===
        elif bear_trend_1d:
            # Primary: Bounce to EMA21 with RSI confirmation
            if price_near_ema21_short and rsi_bounce_short and below_200:
                new_signal = -SIZE_BASE
            
            # Secondary: HMA crossover with 1d confirmation
            elif hma_cross_short and bear_trend_1d and st_bearish:
                new_signal = -SIZE_BASE
            
            # Tertiary: RSI overbought rejection in downtrend
            elif rsi_overbought and bear_trend_12h and zscore_neutral:
                new_signal = -SIZE_HALF
            
            # Momentum: Lower high with trend
            elif lower_high and bear_trend_12h and rsi[i] < 55:
                new_signal = -SIZE_HALF
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Long position stoploss
        if position_side > 0 and entry_price > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        # Short position stoploss
        if position_side < 0 and entry_price > 0:
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            if close[i] > trailing_stop:
                new_signal = 0.0
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals
```

## Last Updated
2026-03-22 09:50
