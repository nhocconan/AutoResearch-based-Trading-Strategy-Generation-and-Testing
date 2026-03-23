# Strategy: daily_ema_rsi_weekly_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.142 | -17.9% | -28.7% | 179 | FAIL |
| ETHUSDT | -1.177 | -29.7% | -36.1% | 181 | FAIL |
| SOLUSDT | 0.029 | +19.0% | -21.0% | 141 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.831 | +18.1% | -6.9% | 34 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #006: Daily EMA Crossover + Weekly Trend Filter + RSI Momentum
Hypothesis: Daily timeframe captures major moves with less noise than intraday.
Weekly EMA provides major trend direction (bull/bear regime).
Daily EMA crossover (8/21) gives entry signals with RSI confirmation.
ATR-based stoploss protects against 2022-style crashes.
Position sizing capped at 0.30 to limit drawdown during bear markets.
This should generate 20-40 trades/year with better risk-adjusted returns.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "daily_ema_rsi_weekly_v1"
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

def calculate_ema(close, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

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

def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD indicator."""
    ema_fast = pd.Series(close).ewm(span=fast, min_periods=fast, adjust=False).mean().values
    ema_slow = pd.Series(close).ewm(span=slow, min_periods=slow, adjust=False).mean().values
    macd_line = ema_fast - ema_slow
    macd_signal = pd.Series(macd_line).ewm(span=signal, min_periods=signal, adjust=False).mean().values
    macd_hist = macd_line - macd_signal
    return macd_line, macd_signal, macd_hist

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return wma3.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load weekly HTF data ONCE before loop (Rule 1)
    df_1w = get_htf_data(prices, '1w')
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate daily indicators
    atr = calculate_atr(high, low, close, 14)
    ema_fast = calculate_ema(close, 8)
    ema_slow = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    rsi = calculate_rsi(close, 14)
    macd_line, macd_signal, macd_hist = calculate_macd(close, 12, 26, 9)
    
    # Volume SMA for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_sma = np.nan_to_num(vol_sma, nan=0.0)
    
    signals = np.zeros(n)
    SIZE = 0.30
    HALF_SIZE = 0.15
    
    # Track positions for stoploss
    entry_price = np.zeros(n)
    position_side = 0
    highest_price = np.zeros(n)
    lowest_price = np.zeros(n)
    
    for i in range(100, n):
        # Weekly trend filter (major regime)
        weekly_bullish = hma_1w_aligned[i] > 0 and close[i] > hma_1w_aligned[i]
        weekly_bearish = hma_1w_aligned[i] > 0 and close[i] < hma_1w_aligned[i]
        
        # Daily EMA crossover
        ema_cross_long = ema_fast[i] > ema_slow[i] and ema_fast[i-1] <= ema_slow[i-1]
        ema_cross_short = ema_fast[i] < ema_slow[i] and ema_fast[i-1] >= ema_slow[i-1]
        ema_trend_long = ema_fast[i] > ema_slow[i] and close[i] > ema_50[i]
        ema_trend_short = ema_fast[i] < ema_slow[i] and close[i] < ema_50[i]
        
        # RSI momentum confirmation
        rsi_long = rsi[i] > 45 and rsi[i] < 70  # Not overbought
        rsi_short = rsi[i] < 55 and rsi[i] > 30  # Not oversold
        rsi_momentum_long = rsi[i] > rsi[i-1] if i > 0 else False
        rsi_momentum_short = rsi[i] < rsi[i-1] if i > 0 else False
        
        # MACD confirmation
        macd_long = macd_hist[i] > 0 and macd_hist[i] > macd_hist[i-1] if i > 0 else False
        macd_short = macd_hist[i] < 0 and macd_hist[i] < macd_hist[i-1] if i > 0 else False
        
        # Volume confirmation
        vol_confirm = volume[i] > vol_sma[i] * 0.8 if vol_sma[i] > 0 else True
        
        # Entry logic - relaxed conditions to ensure trades
        new_signal = 0.0
        
        # Long entry: weekly bullish + EMA trend + RSI ok + volume
        if weekly_bullish and ema_trend_long and rsi_long and vol_confirm:
            new_signal = SIZE
        # Also enter on EMA crossover with weekly support
        elif weekly_bullish and ema_cross_long and rsi[i] > 40:
            new_signal = SIZE
        # Short entry: weekly bearish + EMA trend + RSI ok
        elif weekly_bearish and ema_trend_short and rsi_short and vol_confirm:
            new_signal = -SIZE
        # Also enter on EMA crossover with weekly resistance
        elif weekly_bearish and ema_cross_short and rsi[i] < 60:
            new_signal = -SIZE
        
        # Stoploss logic (Rule 6) - ATR based
        if position_side > 0 and entry_price[i-1] > 0:
            stop_loss = entry_price[i-1] - 2.5 * atr[i]
            if close[i] < stop_loss:
                new_signal = 0.0  # Stoploss hit
            # Trail stop for longs - take partial profit
            elif close[i] > entry_price[i-1] + 3.0 * atr[i]:
                if new_signal == 0:  # Only reduce if not already exiting
                    new_signal = HALF_SIZE
        
        if position_side < 0 and entry_price[i-1] > 0:
            stop_loss = entry_price[i-1] + 2.5 * atr[i]
            if close[i] > stop_loss:
                new_signal = 0.0  # Stoploss hit
            # Trail stop for shorts - take partial profit
            elif close[i] < entry_price[i-1] - 3.0 * atr[i]:
                if new_signal == 0:  # Only reduce if not already exiting
                    new_signal = -HALF_SIZE
        
        # Update position tracking
        if new_signal != 0 and position_side == 0:
            entry_price[i] = close[i]
            position_side = np.sign(new_signal)
            highest_price[i] = close[i]
            lowest_price[i] = close[i]
        elif new_signal != 0 and position_side != 0:
            if np.sign(new_signal) != position_side:
                entry_price[i] = close[i]
                position_side = np.sign(new_signal)
            highest_price[i] = max(highest_price[i-1], close[i])
            lowest_price[i] = min(lowest_price[i-1], close[i])
        else:
            entry_price[i] = entry_price[i-1] if i > 0 else 0
            highest_price[i] = highest_price[i-1] if i > 0 else close[i]
            lowest_price[i] = lowest_price[i-1] if i > 0 else close[i]
            if position_side != 0 and new_signal == 0:
                position_side = 0  # Position closed
        
        signals[i] = new_signal
    
    return signals
```

## Last Updated
2026-03-22 00:33
