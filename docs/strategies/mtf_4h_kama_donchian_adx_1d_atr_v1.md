# Strategy: mtf_4h_kama_donchian_adx_1d_atr_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.508 | -15.8% | -32.0% | 965 | FAIL |
| ETHUSDT | -0.044 | +8.3% | -28.7% | 1032 | FAIL |
| SOLUSDT | 0.589 | +111.3% | -33.0% | 924 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.660 | +23.2% | -21.0% | 380 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #739: 4h Primary + 1d HTF — KAMA Adaptive Trend + Donchian Breakout + ADX Filter

Hypothesis: After 495 failed strategies, clear patterns emerge:
1. Complex regime detection (Chop + CRSI) = 0 trades or negative Sharpe (#727-735)
2. Simple KAMA + Donchian + ADX on 1d got Sharpe=0.234 (#737) — promising template
3. 12h Donchian + HMA 1d got +53.9% return but Sharpe=-0.009 (#736) — needs better exits
4. Current best uses 4h triple regime (Sharpe=0.612) — I'll use simpler 4h logic

Strategy design:
1. 1d HMA(21) for primary trend bias (proven in best strategies)
2. 4h KAMA(14) for adaptive trend following (adjusts to volatility)
3. 4h Donchian(20) breakout for entries (simple, generates trades)
4. 4h ADX(14) > 20 filter to ensure trending conditions (avoid chop)
5. 4h RSI(14) loose filter (35-65) for timing
6. ATR(14) trailing stop 2.5x for risk management
7. Discrete signals: 0.0, ±0.25, ±0.30

Key differences from failed experiments:
- NO Choppiness Index (caused 0 trades in 6+ experiments)
- NO complex CRSI regime switching (failed repeatedly)
- Simple ADX > 20 threshold (not > 40 which rarely triggers)
- Loose RSI filters to ensure trade frequency
- Clear hold logic to maintain positions through trends

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 4h (target 20-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_donchian_adx_1d_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=14, fast=2, slow=30):
    """Kaufman Adaptive Moving Average - adjusts smoothing based on volatility."""
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < period + slow:
        return kama
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        price_change = np.abs(close[i] - close[i - period])
        volatility = np.sum(np.abs(np.diff(close[i-period:i+1])))
        if volatility > 1e-10:
            er[i] = price_change / volatility
        else:
            er[i] = 0
    
    # Calculate Smoothing Constant (SC)
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    
    # Initialize KAMA
    kama[period] = close[period]
    
    # Calculate KAMA
    for i in range(period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_hma(series, period):
    """Hull Moving Average - smoother and more responsive than EMA."""
    if len(series) < period:
        return np.full(len(series), np.nan)
    
    wma1 = pd.Series(series).ewm(span=period//2, min_periods=period//2, adjust=False).mean().values
    wma2 = pd.Series(series).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    hma_raw = 2 * wma1 - wma2
    hma = pd.Series(hma_raw).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean().values
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Donchian Channel for breakout detection."""
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - measures trend strength."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2 + 1:
        return adx
    
    # Calculate True Range and Directional Movement
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(0, high[i] - high[i-1])
        else:
            plus_dm[i] = 0
            
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(0, low[i-1] - low[i])
        else:
            minus_dm[i] = 0
    
    # Smooth TR and DM
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_di = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate DI and DX
    with np.errstate(divide='ignore', invalid='ignore'):
        plus_di = 100 * plus_di / (atr + 1e-10)
        minus_di = 100 * minus_di / (atr + 1e-10)
        di_sum = plus_di + minus_di
        dx = 100 * np.abs(plus_di - minus_di) / (di_sum + 1e-10)
    
    # Calculate ADX
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_sma(close, period):
    """Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (4h) indicators
    kama_4h = calculate_kama(close, period=14)
    rsi_4h = calculate_rsi(close, period=14)
    atr_4h = calculate_atr(high, low, close, period=14)
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    adx_4h = calculate_adx(high, low, close, period=14)
    sma_50 = calculate_sma(close, period=50)
    sma_200 = calculate_sma(close, period=200)
    
    # Calculate and align HTF HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(300, n):  # Need buffer for all indicators + HTF alignment
        # Skip if indicators not ready
        if np.isnan(rsi_4h[i]) or np.isnan(atr_4h[i]) or atr_4h[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(sma_50[i]) or np.isnan(sma_200[i]):
            continue
        if np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
            continue
        if np.isnan(kama_4h[i]) or np.isnan(adx_4h[i]):
            continue
        
        # === TREND BIAS (1d HTF HMA) ===
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # === KAMA TREND (4h) ===
        kama_bullish = close[i] > kama_4h[i]
        kama_bearish = close[i] < kama_4h[i]
        
        # === ADX FILTER (trending market) ===
        trending_market = adx_4h[i] > 20  # Loose threshold to ensure trades
        
        # === RSI FILTERS (loose to ensure trades) ===
        rsi_ok_long = rsi_4h[i] < 70 and rsi_4h[i] > 30
        rsi_ok_short = rsi_4h[i] < 70 and rsi_4h[i] > 30
        
        # === SMA FILTERS ===
        above_sma50 = close[i] > sma_50[i]
        above_sma200 = close[i] > sma_200[i]
        below_sma50 = close[i] < sma_50[i]
        below_sma200 = close[i] < sma_200[i]
        
        desired_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        long_signal = False
        
        # Path 1: Donchian breakout + 1d bullish + ADX trending + KAMA bullish
        if close[i] > donch_upper[i-1] and trend_1d_bullish and trending_market and kama_bullish:
            long_signal = True
        
        # Path 2: Price > KAMA + Price > SMA50 + 1d bullish + ADX > 20
        if kama_bullish and above_sma50 and trend_1d_bullish and adx_4h[i] > 20:
            long_signal = True
        
        # Path 3: Strong trend (above SMA50/200) + 1d bullish + RSI ok
        if above_sma50 and above_sma200 and trend_1d_bullish and rsi_ok_long:
            long_signal = True
        
        # Path 4: KAMA bullish + RSI momentum (40-60) + 1d bullish
        if kama_bullish and 40 < rsi_4h[i] < 60 and trend_1d_bullish:
            long_signal = True
        
        if long_signal:
            desired_signal = BASE_SIZE
        
        # === SHORT ENTRY CONDITIONS ===
        short_signal = False
        
        # Path 1: Donchian breakdown + 1d bearish + ADX trending + KAMA bearish
        if close[i] < donch_lower[i-1] and trend_1d_bearish and trending_market and kama_bearish:
            short_signal = True
        
        # Path 2: Price < KAMA + Price < SMA50 + 1d bearish + ADX > 20
        if kama_bearish and below_sma50 and trend_1d_bearish and adx_4h[i] > 20:
            short_signal = True
        
        # Path 3: Strong downtrend (below SMA50/200) + 1d bearish + RSI ok
        if below_sma50 and below_sma200 and trend_1d_bearish and rsi_ok_short:
            short_signal = True
        
        # Path 4: KAMA bearish + RSI momentum (40-60) + 1d bearish
        if kama_bearish and 40 < rsi_4h[i] < 60 and trend_1d_bearish:
            short_signal = True
        
        if short_signal:
            desired_signal = -BASE_SIZE
        
        # === CONFLICT RESOLUTION ===
        # If both long and short signals, go with 1d HMA trend
        if long_signal and short_signal:
            if trend_1d_bullish:
                desired_signal = BASE_SIZE
            elif trend_1d_bearish:
                desired_signal = -BASE_SIZE
            else:
                desired_signal = 0.0
        
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
        
        # === HOLD LOGIC — Maintain position if trend intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if 1d HMA still bullish and KAMA still bullish
                if trend_1d_bullish and kama_bullish:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if 1d HMA still bearish and KAMA still bearish
                if trend_1d_bearish and kama_bearish:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if 1d trend reverses or KAMA reverses
            if trend_1d_bearish or kama_bearish:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 1d trend reverses or KAMA reverses
            if trend_1d_bullish or kama_bullish:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
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
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals
```

## Last Updated
2026-03-23 13:50
