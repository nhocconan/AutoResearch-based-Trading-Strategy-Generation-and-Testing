# Strategy: mtf_6h_dual_htf_regime_rsi_donchian_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.508 | -11.0% | -25.5% | 690 | FAIL |
| ETHUSDT | 0.008 | +15.6% | -25.9% | 698 | PASS |
| SOLUSDT | 0.958 | +196.2% | -19.8% | 667 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.317 | +12.0% | -17.1% | 219 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #435: 6h Primary + 12h/1d HTF — Dual HTF Bias + Simplified Regime

Hypothesis: 6h timeframe sits between 4h and 12h - should capture multi-day trends
without the whipsaw of lower TFs. Recent 6h failures (#427, #431) show:
- HMA+RSI pullback alone fails (too few signals)
- Weekly pivot filters are too restrictive (0 trades)
- Vol spike strategies fail on 6h

New approach:
1. DUAL HTF BIAS: Both 12h AND 1d must agree for trend entries (reduces false signals)
2. SIMPLER REGIME: ADX > 18 (not 20) + BB Width percentile (more trades qualify)
3. LOOSER RSI: 40/60 thresholds (not 35/65) for mean reversion entries
4. DONCHIAN BREAKOUT: Add 20-bar Donchian as alternative trend entry (proven on HTF)
5. TIGHTER STOP: 2.0x ATR (not 2.5x) to preserve capital in 2022-style crashes

Entry Logic:
- Trending Long: 12h HMA bull + 1d HMA bull + (HMA cross OR Donchian breakout)
- Trending Short: 12h HMA bear + 1d HMA bear + (HMA cross OR Donchian breakdown)
- Choppy Long: RSI < 40 + price > SMA100 (just 2 conditions)
- Choppy Short: RSI > 60 + price < SMA100 (just 2 conditions)

Target: Sharpe>0.45, DD>-35%, trades>=60 train (15/year), trades>=10 test
Timeframe: 6h (NEW - zero prior experiments with this exact setup)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_dual_htf_regime_rsi_donchian_v1"
timeframe = "6h"
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    return upper, lower

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
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA for trend bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (6h) indicators
    hma_6h = calculate_hma(close, period=21)
    hma_6h_fast = calculate_hma(close, period=10)
    atr = calculate_atr(high, low, close, period=14)
    adx = calculate_adx(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    bb_upper, bb_lower, bb_width = calculate_bollinger(close, period=20, std_dev=2.0)
    sma_100 = calculate_sma(close, 100)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
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
        
        if np.isnan(hma_6h[i]) or np.isnan(rsi[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(sma_100[i]) or np.isnan(donchian_upper[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (SIMPLIFIED: ADX + BB Width) ===
        # Trending: ADX > 18 AND BB Width < 50th percentile
        # Choppy: ADX < 18 AND BB Width > 70th percentile
        # Lower ADX threshold (18 vs 20) = more trades qualify
        
        bb_pr = bb_width_pr[i] if not np.isnan(bb_width_pr[i]) else 50.0
        
        is_trending = adx[i] > 18.0 and bb_pr < 50.0
        is_choppy = adx[i] < 18.0 and bb_pr > 70.0
        
        if is_trending:
            current_regime = 1
        elif is_choppy:
            current_regime = 2
        else:
            current_regime = prev_regime
        
        prev_regime = current_regime
        
        # === DUAL HTF BIAS (12h + 1d must agree) ===
        htf_12h_bull = close[i] > hma_12h_aligned[i]
        htf_12h_bear = close[i] < hma_12h_aligned[i]
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # Both HTF must agree for strong trend signal
        htf_both_bull = htf_12h_bull and htf_1d_bull
        htf_both_bear = htf_12h_bear and htf_1d_bear
        
        # === 6h HMA TREND ===
        hma_bull = close[i] > hma_6h[i]
        hma_bear = close[i] < hma_6h[i]
        
        # === HMA CROSSOVER ===
        hma_cross_long = False
        hma_cross_short = False
        if i > 0 and not np.isnan(hma_6h_fast[i]) and not np.isnan(hma_6h_fast[i-1]):
            if not np.isnan(hma_6h[i]) and not np.isnan(hma_6h[i-1]):
                if hma_6h_fast[i-1] <= hma_6h[i-1] and hma_6h_fast[i] > hma_6h[i]:
                    hma_cross_long = True
                if hma_6h_fast[i-1] >= hma_6h[i-1] and hma_6h_fast[i] < hma_6h[i]:
                    hma_cross_short = True
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        donchian_breakdown_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === SMA100 FILTER ===
        above_sma100 = close[i] > sma_100[i]
        below_sma100 = close[i] < sma_100[i]
        
        # === RSI EXTREMES (LOOSENED: 40/60 instead of 35/65) ===
        rsi_oversold = rsi[i] < 40.0
        rsi_overbought = rsi[i] > 60.0
        
        # === BB TOUCH ===
        touch_lower = close[i] <= bb_lower[i] if not np.isnan(bb_lower[i]) else False
        touch_upper = close[i] >= bb_upper[i] if not np.isnan(bb_upper[i]) else False
        
        # === ENTRY LOGIC (LOOSE - max 2-3 conditions) ===
        desired_signal = 0.0
        
        # REGIME 1: TRENDING (breakout + dual HTF alignment)
        if current_regime == 1:
            # Long: Dual HTF bull + (HMA bull OR Donchian breakout OR HMA cross)
            if htf_both_bull:
                if hma_bull or donchian_breakout_long or hma_cross_long:
                    desired_signal = SIZE_STRONG
            
            # Short: Dual HTF bear + (HMA bear OR Donchian breakdown OR HMA cross)
            elif htf_both_bear:
                if hma_bear or donchian_breakdown_short or hma_cross_short:
                    desired_signal = -SIZE_STRONG
        
        # REGIME 2: CHOPPY (RSI mean reversion - VERY LOOSE)
        elif current_regime == 2:
            # Long: RSI < 40 + above SMA100 (just 2 conditions!)
            if rsi_oversold and above_sma100:
                desired_signal = SIZE_BASE
            
            # Short: RSI > 60 + below SMA100 (just 2 conditions!)
            elif rsi_overbought and below_sma100:
                desired_signal = -SIZE_BASE
            
            # Extra: BB touch for additional mean reversion entries
            elif touch_lower and rsi[i] < 50.0:
                desired_signal = SIZE_BASE
            elif touch_upper and rsi[i] > 50.0:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.0x ATR from entry) ===
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
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
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
2026-03-24 14:35
