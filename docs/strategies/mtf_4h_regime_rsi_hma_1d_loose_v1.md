# Strategy: mtf_4h_regime_rsi_hma_1d_loose_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.013 | -32.3% | -42.0% | 1023 | FAIL |
| ETHUSDT | -0.079 | +8.7% | -23.2% | 1064 | FAIL |
| SOLUSDT | 0.405 | +63.4% | -35.7% | 1024 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.647 | +20.3% | -18.9% | 339 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #428: 4h Primary + 1d HTF — Simplified Regime + Loose Entries

Hypothesis: Recent failures (#416-#427) show Sharpe=0.000 due to OVERLY STRICT entries.
This version LOOSENS conditions dramatically while keeping proven regime detection.

Key changes from failed experiments:
1. RSI thresholds: 35/65 (not 25/75) — triggers more mean reversion trades
2. ADX threshold: 20 (not 25) — more regimes qualify as "trending"
3. Max 2 confluence filters per entry (not 4-5)
4. Remove volume confirmation requirement (blocks too many valid entries)
5. Simplified regime: ADX + BB Width only (no Choppiness complexity)

Regime Detection (SIMPLE):
- ADX > 20 + BB Width < 30th percentile = trending → HMA breakout
- ADX < 20 + BB Width > 70th percentile = choppy → RSI mean revert
- Otherwise = neutral → flat or reduce size

Entry Logic (LOOSE):
- Trending Long: HMA(21) bull + 1d HMA bull + price > HMA (3 conditions max)
- Trending Short: HMA(21) bear + 1d HMA bear + price < HMA (3 conditions max)
- Choppy Long: RSI < 35 + price > SMA100 (just 2 conditions!)
- Choppy Short: RSI > 65 + price < SMA100 (just 2 conditions!)

Position sizing: 0.25 base, 0.30 when HTF aligned
Stoploss: 2.5x ATR(14) from entry price

Target: Sharpe>0.45, DD>-35%, trades>=80 train (20/year), trades>=15 test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_regime_rsi_hma_1d_loose_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength"""
    n = len(close)
    if n < period * 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up = high[i] - high[i-1]
        down = low[i-1] - low[i]
        if up > down and up > 0:
            plus_dm[i] = up
        if down > up and down > 0:
            minus_dm[i] = down
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 1e-10:
            plus_di[i] = 100.0 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
            minus_di[i] = 100.0 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
    
    dx = np.zeros(n)
    dx[:] = np.nan
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

def calculate_bollinger(close, period=20, std_dev=2.0):
    """Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    
    # Band width as % of price
    bw = np.zeros(n)
    bw[:] = np.nan
    for i in range(period, n):
        if sma[i] > 1e-10:
            bw[i] = 100.0 * (upper[i] - lower[i]) / sma[i]
    
    return upper, lower, bw

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def calculate_percentile_rank(values, period=100):
    """Percentile rank for BB Width regime detection"""
    n = len(values)
    pr = np.zeros(n)
    pr[:] = np.nan
    
    for i in range(period, n):
        window = values[i-period+1:i+1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            pr[i] = np.sum(valid < values[i]) / len(valid) * 100.0
    
    return pr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    hma_4h = calculate_hma(close, period=21)
    hma_4h_fast = calculate_hma(close, period=10)
    atr = calculate_atr(high, low, close, period=14)
    adx = calculate_adx(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    bb_upper, bb_lower, bb_width = calculate_bollinger(close, period=20, std_dev=2.0)
    sma_100 = calculate_sma(close, 100)
    
    # BB Width percentile for regime detection
    bb_width_pr = calculate_percentile_rank(bb_width, period=100)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Regime memory for hysteresis
    prev_regime = 0  # 0=unknown, 1=trending, 2=choppy
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h[i]) or np.isnan(rsi[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(sma_100[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (SIMPLIFIED: ADX + BB Width) ===
        # Trending: ADX > 20 AND BB Width < 50th percentile (narrow bands = trend building)
        # Choppy: ADX < 20 AND BB Width > 70th percentile (wide bands = range)
        
        bb_pr = bb_width_pr[i] if not np.isnan(bb_width_pr[i]) else 50.0
        
        is_trending = adx[i] > 20.0 and bb_pr < 50.0
        is_choppy = adx[i] < 20.0 and bb_pr > 70.0
        
        if is_trending:
            current_regime = 1
        elif is_choppy:
            current_regime = 2
        else:
            current_regime = prev_regime
        
        prev_regime = current_regime
        
        # === HTF BIAS (1d) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === 4h HMA TREND ===
        hma_bull = close[i] > hma_4h[i]
        hma_bear = close[i] < hma_4h[i]
        
        # === HMA CROSSOVER ===
        hma_cross_long = False
        hma_cross_short = False
        if i > 0 and not np.isnan(hma_4h_fast[i]) and not np.isnan(hma_4h_fast[i-1]):
            if not np.isnan(hma_4h[i]) and not np.isnan(hma_4h[i-1]):
                if hma_4h_fast[i-1] <= hma_4h[i-1] and hma_4h_fast[i] > hma_4h[i]:
                    hma_cross_long = True
                if hma_4h_fast[i-1] >= hma_4h[i-1] and hma_4h_fast[i] < hma_4h[i]:
                    hma_cross_short = True
        
        # === SMA100 FILTER (looser than SMA200) ===
        above_sma100 = close[i] > sma_100[i]
        below_sma100 = close[i] < sma_100[i]
        
        # === RSI EXTREMES (LOOSENED: 35/65 instead of 25/75) ===
        rsi_oversold = rsi[i] < 35.0
        rsi_overbought = rsi[i] > 65.0
        
        # === BB TOUCH ===
        touch_lower = close[i] <= bb_lower[i] if not np.isnan(bb_lower[i]) else False
        touch_upper = close[i] >= bb_upper[i] if not np.isnan(bb_upper[i]) else False
        
        # === ENTRY LOGIC (LOOSE - max 2-3 conditions) ===
        desired_signal = 0.0
        
        # REGIME 1: TRENDING (breakout + trend alignment)
        if current_regime == 1:
            # Long: HMA bull + 1d bull + (price > HMA OR cross)
            if hma_bull and htf_1d_bull:
                if close[i] > hma_4h[i] or hma_cross_long:
                    desired_signal = SIZE_STRONG
            
            # Short: HMA bear + 1d bear + (price < HMA OR cross)
            elif hma_bear and htf_1d_bear:
                if close[i] < hma_4h[i] or hma_cross_short:
                    desired_signal = -SIZE_STRONG
        
        # REGIME 2: CHOPPY (RSI mean reversion - VERY LOOSE)
        elif current_regime == 2:
            # Long: RSI < 35 + above SMA100 (just 2 conditions!)
            if rsi_oversold and above_sma100:
                desired_signal = SIZE_BASE
            
            # Short: RSI > 65 + below SMA100 (just 2 conditions!)
            elif rsi_overbought and below_sma100:
                desired_signal = -SIZE_BASE
            
            # Extra: BB touch for additional mean reversion entries
            elif touch_lower and rsi[i] < 45.0:
                desired_signal = SIZE_BASE
            elif touch_upper and rsi[i] > 55.0:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                # Set stoploss
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
        
        signals[i] = final_signal
    
    return signals
```

## Last Updated
2026-03-24 14:22
