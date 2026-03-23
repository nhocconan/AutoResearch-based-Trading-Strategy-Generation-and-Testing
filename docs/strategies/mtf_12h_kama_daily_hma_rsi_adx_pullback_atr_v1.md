# Strategy: mtf_12h_kama_daily_hma_rsi_adx_pullback_atr_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.268 | -2.6% | -28.7% | 400 | FAIL |
| ETHUSDT | -0.341 | -15.0% | -28.6% | 402 | FAIL |
| SOLUSDT | 0.570 | +106.5% | -30.0% | 361 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.048 | +4.1% | -23.1% | 119 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #437: 12h KAMA Adaptive Trend + Daily HMA Bias + RSI Pullback + ADX Filter
Hypothesis: KAMA adapts to volatility better than EMA/HMA, reducing whipsaws in choppy markets.
12h timeframe captures medium-term trends with less noise than intraday. Daily HMA provides
higher timeframe bias. RSI pullback entries (not extremes) catch trend continuations.
ADX filter (>20) ensures we only trade when there's actual trend momentum.
Multiple entry paths ensure >=10 trades per symbol while maintaining quality.
Position size: 0.30 discrete, stoploss 2.5*ATR for 12h timeframe.
Timeframe: 12h (REQUIRED), HTF: 1d for trend bias via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_daily_hma_rsi_adx_pullback_atr_v1"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average.
    KAMA adapts to market noise - moves fast in trends, slow in chop.
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio (ER)
    for i in range(er_period, n):
        change = np.abs(close[i] - close[i - er_period])
        volatility = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if volatility > 0:
            er = change / volatility
        else:
            er = 0.0
        
        # Calculate smoothing constant
        fast_sc = 2 / (fast_period + 1)
        slow_sc = 2 / (slow_period + 1)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        # Initialize KAMA
        if i == er_period:
            kama[i] = close[i]
        else:
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    n = len(close)
    adx = np.zeros(n)
    adx[:] = np.nan
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_dm[i] = max(0, high[i] - high[i-1]) if (high[i] - high[i-1]) > (low[i-1] - low[i]) else 0
        minus_dm[i] = max(0, low[i-1] - low[i]) if (low[i-1] - low[i]) > (high[i] - high[i-1]) else 0
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr * 100
    minus_di = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr * 100
    
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10) * 100
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

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
    kama = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    sma50 = calculate_sma(close, 50)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.30
    SIZE_HALF = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):  # Start after 100 bars for indicators
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(rsi[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(sma50[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(kama[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        # Daily trend bias (long-term direction)
        daily_bullish = close[i] > hma_1d_aligned[i]
        daily_bearish = close[i] < hma_1d_aligned[i]
        
        # 12h trend filter
        kama_bullish = close[i] > kama[i]
        kama_bearish = close[i] < kama[i]
        above_sma50 = close[i] > sma50[i]
        below_sma50 = close[i] < sma50[i]
        
        # ADX trend strength filter (only trade when ADX > 20)
        trend_strength = adx[i] > 20
        
        # RSI pullback conditions (not extremes - catch trend continuations)
        rsi_pullback_long = rsi[i] > 40 and rsi[i] < 55
        rsi_pullback_short = rsi[i] > 45 and rsi[i] < 60
        rsi_momentum_long = rsi[i] > 50 and rsi[i] < 70
        rsi_momentum_short = rsi[i] > 30 and rsi[i] < 50
        
        # DI crossover signals
        di_bullish = plus_di[i] > minus_di[i]
        di_bearish = plus_di[i] < minus_di[i]
        
        # KAMA slope (trend direction)
        kama_slope_up = kama[i] > kama[i-1] if not np.isnan(kama[i-1]) else False
        kama_slope_down = kama[i] < kama[i-1] if not np.isnan(kama[i-1]) else False
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths to ensure >=10 trades) ===
        # Path 1: KAMA bullish + Daily bullish + ADX trend + RSI pullback
        if kama_bullish and daily_bullish and trend_strength and rsi_pullback_long:
            new_signal = SIZE_ENTRY
        # Path 2: DI bullish + Daily bullish + Above SMA50 + RSI momentum
        elif di_bullish and daily_bullish and above_sma50 and rsi_momentum_long:
            new_signal = SIZE_ENTRY
        # Path 3: KAMA slope up + Daily bullish + ADX > 25 + RSI > 45
        elif kama_slope_up and daily_bullish and adx[i] > 25 and rsi[i] > 45 and rsi[i] < 65:
            new_signal = SIZE_ENTRY
        # Path 4: Price above KAMA + DI bullish + Daily bullish + RSI 40-60
        elif close[i] > kama[i] and di_bullish and daily_bullish and rsi[i] > 40 and rsi[i] < 60:
            new_signal = SIZE_ENTRY
        # Path 5: Simple trend continuation - Above SMA50 + Daily bullish + KAMA bullish
        elif above_sma50 and daily_bullish and kama_bullish and rsi[i] > 45:
            new_signal = SIZE_ENTRY
        # Path 6: RSI recovery + KAMA support + Daily bullish
        elif rsi[i] > rsi[i-1] and rsi[i-1] < 45 and close[i] > kama[i] and daily_bullish:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths to ensure >=10 trades) ===
        # Path 1: KAMA bearish + Daily bearish + ADX trend + RSI pullback
        if kama_bearish and daily_bearish and trend_strength and rsi_pullback_short:
            new_signal = -SIZE_ENTRY
        # Path 2: DI bearish + Daily bearish + Below SMA50 + RSI momentum
        elif di_bearish and daily_bearish and below_sma50 and rsi_momentum_short:
            new_signal = -SIZE_ENTRY
        # Path 3: KAMA slope down + Daily bearish + ADX > 25 + RSI < 55
        elif kama_slope_down and daily_bearish and adx[i] > 25 and rsi[i] > 35 and rsi[i] < 55:
            new_signal = -SIZE_ENTRY
        # Path 4: Price below KAMA + DI bearish + Daily bearish + RSI 40-60
        elif close[i] < kama[i] and di_bearish and daily_bearish and rsi[i] > 40 and rsi[i] < 60:
            new_signal = -SIZE_ENTRY
        # Path 5: Simple trend continuation - Below SMA50 + Daily bearish + KAMA bearish
        elif below_sma50 and daily_bearish and kama_bearish and rsi[i] < 55:
            new_signal = -SIZE_ENTRY
        # Path 6: RSI rejection + KAMA resistance + Daily bearish
        elif rsi[i] < rsi[i-1] and rsi[i-1] > 55 and close[i] < kama[i] and daily_bearish:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from highest for 12h timeframe)
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
            
            # Calculate trailing stop (2.5*ATR from lowest for 12h timeframe)
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
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
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
2026-03-22 06:30
