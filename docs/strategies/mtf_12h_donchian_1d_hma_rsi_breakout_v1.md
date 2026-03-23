# Strategy: mtf_12h_donchian_1d_hma_rsi_breakout_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.677 | +1.1% | -14.4% | 298 | FAIL |
| ETHUSDT | -1.038 | -18.0% | -25.8% | 363 | FAIL |
| SOLUSDT | 0.536 | +58.2% | -16.4% | 357 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.782 | +16.9% | -8.7% | 116 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #077: 12h Donchian Breakout with 1d HMA Trend Filter + RSI Confirmation
Hypothesis: 12h timeframe is slow enough for trend-following breakouts to work without whipsaw.
Donchian(20) breakouts capture sustained moves, 1d HMA provides trend bias, RSI avoids extremes.
Key insight: Previous 12h strategies failed due to too many conflicting filters or mean-reversion focus.
12h is ideal for breakout strategies - fewer false signals than lower TFs, more trades than 1d.
This strategy uses:
- Donchian(20) breakout for entry signals (price breaks 20-bar high/low)
- 1d HMA(21) for trend bias (long only above, short only below)
- RSI(14) filter (30-70 range, avoid extreme overbought/oversold entries)
- ATR(14) trailing stop at 2.5x for risk management
- Conservative position sizing (0.25-0.30 discrete levels)
Why this might work: Donchian breakouts work well on slower timeframes (12h+).
1d HMA provides gentle trend filter without killing trade frequency.
RSI ensures we're not buying tops or selling bottoms. Fewer filters = more trades.
Timeframe: 12h (REQUIRED), HTF: 1d via mtf_data helper (call ONCE before loop).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_1d_hma_rsi_breakout_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_donchian(high, low, period=20):
    """
    Calculate Donchian Channel.
    Returns: upper_band (highest high), lower_band (lowest low)
    """
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

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
    
    # Donchian channels
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(ema_21[i]) or np.isnan(donchian_upper[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 1d HMA = intermediate trend bias
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        # Long: price breaks above Donchian upper
        donchian_breakout_long = close[i] > donchian_upper[i - 1] if i > 0 else False
        # Short: price breaks below Donchian lower
        donchian_breakout_short = close[i] < donchian_lower[i - 1] if i > 0 else False
        
        # === EMA ALIGNMENT ===
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        
        # === RSI FILTER (avoid extreme entries) ===
        # For longs: RSI not overbought (< 70), preferably > 40
        rsi_ok_long = 35 <= rsi[i] <= 70
        # For shorts: RSI not oversold (> 30), preferably < 60
        rsi_ok_short = 30 <= rsi[i] <= 65
        
        # RSI momentum confirmation
        rsi_momentum_long = rsi[i] > 45
        rsi_momentum_short = rsi[i] < 55
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        
        # Path 1: Donchian breakout + 1d trend bullish + RSI OK
        if donchian_breakout_long and bull_trend_1d:
            if rsi_ok_long:
                if ema_bullish and rsi_momentum_long:
                    new_signal = SIZE_STRONG
                else:
                    new_signal = SIZE_BASE
        
        # Path 2: Price above 1d HMA + EMA bullish + RSI momentum (trend continuation)
        if bull_trend_1d and ema_bullish:
            if rsi[i] > 50 and rsi[i] < 65:
                if close[i] > ema_21[i]:
                    new_signal = SIZE_BASE
        
        # Path 3: Simple breakout with trend confirmation (ensure trades happen)
        if donchian_breakout_long:
            if bull_trend_1d or ema_bullish:
                if rsi[i] > 40 and rsi[i] < 75:
                    new_signal = SIZE_BASE
        
        # === SHORT ENTRY CONDITIONS ===
        
        # Path 1: Donchian breakout + 1d trend bearish + RSI OK
        if donchian_breakout_short and bear_trend_1d:
            if rsi_ok_short:
                if ema_bearish and rsi_momentum_short:
                    new_signal = -SIZE_STRONG
                else:
                    new_signal = -SIZE_BASE
        
        # Path 2: Price below 1d HMA + EMA bearish + RSI momentum (trend continuation)
        if bear_trend_1d and ema_bearish:
            if rsi[i] < 50 and rsi[i] > 35:
                if close[i] < ema_21[i]:
                    new_signal = -SIZE_BASE
        
        # Path 3: Simple breakout with trend confirmation (ensure trades happen)
        if donchian_breakout_short:
            if bear_trend_1d or ema_bearish:
                if rsi[i] > 25 and rsi[i] < 60:
                    new_signal = -SIZE_BASE
        
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
            if lowest_close == 0.0 or close[i] < lowest_close:
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
2026-03-22 11:22
