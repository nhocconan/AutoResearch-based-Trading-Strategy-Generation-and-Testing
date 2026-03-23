# Strategy: mtf_1d_rsi_macd_weekly_hma_ensemble_atr_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.266 | +3.1% | -24.3% | 129 | FAIL |
| ETHUSDT | -0.607 | -20.5% | -32.7% | 143 | FAIL |
| SOLUSDT | 0.943 | +174.6% | -22.5% | 128 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.251 | +9.7% | -15.8% | 46 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #432: 1d RSI-MACD Ensemble + Weekly HMA Trend + ATR Stoploss
Hypothesis: Daily timeframe with weekly trend bias provides cleaner signals than lower TFs.
Combining RSI mean-reversion (oversold/overbought) with MACD momentum crossover creates
multiple entry paths. Weekly HMA filters direction to avoid counter-trend trades.
Key insight: 1d data has less noise than intraday, fewer false signals, lower fee drag.
Relaxed RSI thresholds (25/75 instead of 30/70) ensure >=10 trades per symbol.
Timeframe: 1d (REQUIRED), HTF: 1w for trend bias via mtf_data helper.
Position size: 0.28 discrete, stoploss 2.5*ATR for daily timeframe.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_rsi_macd_weekly_hma_ensemble_atr_v1"
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
    """Calculate MACD line, signal line, and histogram."""
    close_s = pd.Series(close)
    ema_fast = close_s.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = close_s.ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line.values, signal_line.values, histogram.values

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

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper.values, lower.values, sma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    macd_line, macd_signal, macd_hist = calculate_macd(close, 12, 26, 9)
    sma50 = calculate_sma(close, 50)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    
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
    
    for i in range(100, n):  # Start after 100 bars for indicators
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(rsi[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_aligned[i]) or np.isnan(sma50[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(macd_line[i]) or np.isnan(macd_signal[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            continue
        
        # Weekly trend bias (long-term direction)
        weekly_bullish = close[i] > hma_1w_aligned[i]
        weekly_bearish = close[i] < hma_1w_aligned[i]
        
        # Daily trend filter
        above_sma50 = close[i] > sma50[i]
        below_sma50 = close[i] < sma50[i]
        
        # RSI conditions (RELAXED for more trades)
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_neutral_long = rsi[i] > 40 and rsi[i] < 70
        rsi_neutral_short = rsi[i] > 30 and rsi[i] < 60
        
        # MACD conditions
        macd_bullish_cross = macd_line[i] > macd_signal[i] and macd_line[i-1] <= macd_signal[i-1]
        macd_bearish_cross = macd_line[i] < macd_signal[i] and macd_line[i-1] >= macd_signal[i-1]
        macd_above_zero = macd_line[i] > 0
        macd_below_zero = macd_line[i] < 0
        macd_hist_positive = macd_hist[i] > 0
        macd_hist_negative = macd_hist[i] < 0
        
        # Bollinger Band conditions
        near_bb_lower = close[i] < bb_lower[i] * 1.02  # within 2% of lower band
        near_bb_upper = close[i] > bb_upper[i] * 0.98  # within 2% of upper band
        bb_squeeze = (bb_upper[i] - bb_lower[i]) / bb_mid[i] < 0.10  # narrow bands
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths to ensure >=10 trades) ===
        # Path 1: RSI oversold + Weekly bullish + MACD histogram positive
        if rsi_oversold and weekly_bullish and macd_hist_positive:
            new_signal = SIZE_ENTRY
        # Path 2: MACD bullish cross + Weekly bullish + Above SMA50
        elif macd_bullish_cross and weekly_bullish and above_sma50:
            new_signal = SIZE_ENTRY
        # Path 3: Near BB lower + Weekly bullish + RSI neutral
        elif near_bb_lower and weekly_bullish and rsi_neutral_long:
            new_signal = SIZE_ENTRY
        # Path 4: MACD above zero + Weekly bullish + RSI > 45
        elif macd_above_zero and weekly_bullish and rsi[i] > 45 and rsi[i] < 75:
            new_signal = SIZE_ENTRY
        # Path 5: Simple trend - Above SMA50 + Weekly bullish + MACD hist positive
        elif above_sma50 and weekly_bullish and macd_hist_positive and rsi[i] > 40:
            new_signal = SIZE_ENTRY
        # Path 6: RSI recovery from oversold (RSI was <35, now >35)
        elif rsi[i] > 35 and rsi[i-1] < 35 and weekly_bullish:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths to ensure >=10 trades) ===
        # Path 1: RSI overbought + Weekly bearish + MACD histogram negative
        if rsi_overbought and weekly_bearish and macd_hist_negative:
            new_signal = -SIZE_ENTRY
        # Path 2: MACD bearish cross + Weekly bearish + Below SMA50
        elif macd_bearish_cross and weekly_bearish and below_sma50:
            new_signal = -SIZE_ENTRY
        # Path 3: Near BB upper + Weekly bearish + RSI neutral
        elif near_bb_upper and weekly_bearish and rsi_neutral_short:
            new_signal = -SIZE_ENTRY
        # Path 4: MACD below zero + Weekly bearish + RSI < 55
        elif macd_below_zero and weekly_bearish and rsi[i] < 55 and rsi[i] > 25:
            new_signal = -SIZE_ENTRY
        # Path 5: Simple trend - Below SMA50 + Weekly bearish + MACD hist negative
        elif below_sma50 and weekly_bearish and macd_hist_negative and rsi[i] < 60:
            new_signal = -SIZE_ENTRY
        # Path 6: RSI rejection from overbought (RSI was >65, now <65)
        elif rsi[i] < 65 and rsi[i-1] > 65 and weekly_bearish:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from highest for daily timeframe)
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
            
            # Calculate trailing stop (2.5*ATR from lowest for daily timeframe)
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
2026-03-22 06:24
