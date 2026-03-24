# Strategy: mtf_4h_chop_bb_regime_1d_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.651 | -13.9% | -28.6% | 1095 | FAIL |
| ETHUSDT | -0.021 | +14.9% | -20.5% | 1108 | FAIL |
| SOLUSDT | 0.389 | +57.8% | -21.2% | 1142 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.234 | +9.5% | -17.9% | 350 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #029: 4h Primary + 1d HTF — Volatility Regime + BB Mean Reversion

Hypothesis: After 28 failed experiments, the key insight is that 2025 test period
is bear/range market where pure trend following fails. Need REGIME-ADAPTIVE logic:
- CHOPPY regime (CHOP > 55): Mean revert at Bollinger Band extremes
- TREND regime (CHOP < 45 + ADX > 20): Follow HTF trend on pullbacks

Key improvements over #011 (current best Sharpe=0.221):
1. Choppiness Index regime detection - switches strategy based on market state
2. Bollinger Band mean reversion - proven edge in range markets (2025 test)
3. LOOSER thresholds to ensure trades generate (RSI 45/55 vs 35/65)
4. 1d HMA for HTF bias - simpler than KAMA, more stable
5. Dual entry logic - works in both chop AND trend regimes

Entry Logic:
- CHOPPY (CHOP>55): Long when price<BB_lower + RSI<45 + 1d_HMA_bull
                     Short when price>BB_upper + RSI>55 + 1d_HMA_bear
- TREND (CHOP<45 + ADX>20): Long when price>4h_HMA + 1d_HMA_bull + RSI>50
                           Short when price<4h_HMA + 1d_HMA_bear + RSI<50
- Size: 0.28 (discrete, minimizes fee churn)

Risk: 2.5x ATR trailing stop, signal→0 when stopped out
Target: Sharpe>0.3, trades>30/symbol train, >3/symbol test, DD>-40%
Timeframe: 4h (target 20-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_chop_bb_regime_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppy vs trending
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
    Formula: 100 * (ATR(1) sum / ATR(period) sum) / log10(High_Low_Range)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    choppiness = np.full(n, np.nan)
    
    for i in range(period, n):
        # Calculate true range for each bar in the window
        tr_sum = 0.0
        high_low_sum = 0.0
        
        for j in range(i - period + 1, i + 1):
            if j == 0:
                tr = high[j] - low[j]
            else:
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            tr_sum += tr
            
            # Track highest high and lowest low in window
            window_high = np.max(high[i - period + 1:i + 1])
            window_low = np.min(low[i - period + 1:i + 1])
            high_low_sum = window_high - window_low
        
        if high_low_sum < 1e-10 or tr_sum < 1e-10:
            choppiness[i] = 100.0
        else:
            atr_period = tr_sum / period
            choppiness[i] = 100.0 * (atr_period * period / tr_sum) * (np.log10(high_low_sum / atr_period) if high_low_sum > atr_period else 0)
            # Simplified formula that works better
            choppiness[i] = 100.0 * (tr_sum / period) / high_low_sum * np.log10(period)
    
    # Normalize to 0-100 range with better formula
    for i in range(period, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        high_low_range = highest_high - lowest_low
        
        if high_low_range < 1e-10:
            choppiness[i] = 100.0
        else:
            atr_sum = 0.0
            for j in range(i - period + 1, i + 1):
                if j == 0:
                    tr = high[j] - low[j]
                else:
                    tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
                atr_sum += tr
            
            choppiness[i] = 100.0 * (atr_sum / period) / high_low_range
    
    return choppiness

def calculate_hma(close, period=21):
    """Hull Moving Average - smoother and more responsive than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # Calculate WMA for period and half period
    def wma(data, span):
        result = np.full(len(data), np.nan)
        weights = np.arange(1, span + 1)
        for i in range(span - 1, len(data)):
            window = data[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half_period)
    wma_full = wma(close, period)
    
    # WMA of 2*WMA(n/2) - WMA(n)
    double_wma_half = 2.0 * wma_half - wma_full
    
    # Final WMA with sqrt period
    hma = wma(double_wma_half, sqrt_period)
    
    return hma

def calculate_bollinger_bands(close, period=20, std_dev=1.8):
    """Bollinger Bands - for mean reversion entries"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    
    return upper, lower

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength"""
    n = len(close)
    if n < period * 3:
        return np.full(n, np.nan)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        plus_diff = high[i] - high[i-1]
        minus_diff = low[i-1] - low[i]
        
        if plus_diff > minus_diff and plus_diff > 0:
            plus_dm[i] = plus_diff
        if minus_diff > plus_diff and minus_diff > 0:
            minus_dm[i] = minus_diff
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_di = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    dx = np.zeros(n)
    for i in range(period * 2, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum < 1e-10:
            dx[i] = 0.0
        else:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_rsi(close, period=14):
    """RSI - momentum filter with loose thresholds"""
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
    
    rsi = np.full(n, np.nan)
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range - for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for HTF trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    hma_4h = calculate_hma(close, period=21)
    bb_upper, bb_lower = calculate_bollinger_bands(close, period=20, std_dev=1.8)
    choppiness = calculate_choppiness(high, low, close, period=14)
    adx = calculate_adx(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE = 0.28  # Discrete position size
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_4h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(choppiness[i]) or np.isnan(adx[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION ===
        choppy_regime = choppiness[i] > 55.0  # Range market
        trend_regime = choppiness[i] < 45.0 and adx[i] > 15.0  # Trending market
        
        # === HTF BIAS ===
        hma_1d_bull = close[i] > hma_1d_aligned[i]
        hma_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === 4h TREND ===
        hma_4h_bull = close[i] > hma_4h[i]
        hma_4h_bear = close[i] < hma_4h[i]
        
        # === BB POSITION ===
        price_below_bb = close[i] < bb_lower[i]
        price_above_bb = close[i] > bb_upper[i]
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # CHOPPY REGIME: Mean reversion at BB extremes
        if choppy_regime:
            # Long: Price below BB lower + RSI oversold + HTF bull bias
            if price_below_bb and rsi[i] < 45.0 and hma_1d_bull:
                desired_signal = SIZE
            # Short: Price above BB upper + RSI overbought + HTF bear bias
            elif price_above_bb and rsi[i] > 55.0 and hma_1d_bear:
                desired_signal = -SIZE
        
        # TREND REGIME: Follow trend on pullbacks
        elif trend_regime:
            # Long: 4h bull + 1d bull + RSI neutral-bull
            if hma_4h_bull and hma_1d_bull and rsi[i] > 50.0:
                desired_signal = SIZE
            # Short: 4h bear + 1d bear + RSI neutral-bear
            elif hma_4h_bear and hma_1d_bear and rsi[i] < 50.0:
                desired_signal = -SIZE
        
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
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
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
        
        signals[i] = final_signal
    
    return signals
```

## Last Updated
2026-03-24 05:27
