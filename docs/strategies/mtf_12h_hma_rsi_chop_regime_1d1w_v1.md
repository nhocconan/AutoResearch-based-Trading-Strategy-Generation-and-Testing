# Strategy: mtf_12h_hma_rsi_chop_regime_1d1w_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.996 | -7.8% | -21.0% | 569 | FAIL |
| ETHUSDT | -1.181 | -20.1% | -27.8% | 581 | FAIL |
| SOLUSDT | 0.396 | +43.8% | -12.4% | 544 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.213 | +8.9% | -14.3% | 200 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #624: 12h Primary + 1d/1w HTF — HMA Trend + RSI Pullback + Choppiness Filter

Hypothesis: 12h timeframe balances trade frequency (20-50/year) with signal quality.
Higher timeframes have proven more robust than lower TFs which generate 0 trades.

Key improvements over failed 6h strategies:
1. 12h instead of 6h - less noise, fewer whipsaws, proven to work better
2. RSI pullback instead of Fisher - more reliable for trend entries
3. Simpler regime filter - Choppiness > 55 = reduce size (don't block entries)
4. LOOSE entry conditions - ensure we generate trades (avoid 0-trade failure)
5. HMA trend following - proven to work on higher timeframes

Strategy logic:
1. 1w HMA(21) = macro trend bias (slow filter)
2. 1d HMA(21) = medium trend bias (primary filter)
3. 12h HMA(21) = local trend confirmation
4. 12h RSI(14) = pullback entry (wide zone 35-65 to ensure trades)
5. 12h Choppiness(14) = size modifier (>55 = reduce size 50%)
6. 12h ATR(14) = stoploss (2.5*ATR trailing)

Entry conditions (LOOSE to ensure trades):
- LONG: close > 1d HMA AND RSI(14) < 65 (pullback in uptrend)
- SHORT: close < 1d HMA AND RSI(14) > 35 (bounce in downtrend)
- 1w HMA confirms macro direction (optional boost)
- Choppiness only reduces size, doesn't block entries

Target: Sharpe>0.40, trades>=30 train, trades>=3 test
Timeframe: 12h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_rsi_chop_regime_1d1w_v1"
timeframe = "12h"
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

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - measures market choppy vs trending"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        atr_sum = np.nansum(atr[i-period+1:i+1])
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMAs
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 12h indicators
    hma_12h = calculate_hma(close, period=21)
    rsi = calculate_rsi(close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(chop[i]) or np.isnan(hma_12h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d primary, 1w confirmation) ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # 1w macro boost (optional - increases confidence)
        macro_bull = close[i] > hma_1w_aligned[i]
        macro_bear = close[i] < hma_1w_aligned[i]
        
        # === LOCAL TREND ===
        local_bull = close[i] > hma_12h[i]
        local_bear = close[i] < hma_12h[i]
        
        # === CHOPPINESS REGIME ===
        is_choppy = chop[i] > 55.0
        
        # === RSI PULLBACK ZONES (WIDE to ensure trades) ===
        # Long: RSI not overbought (< 65) in uptrend
        # Short: RSI not oversold (> 35) in downtrend
        rsi_ok_long = rsi[i] < 65.0
        rsi_ok_short = rsi[i] > 35.0
        
        # RSI momentum (improving for long, weakening for short)
        rsi_momentum_long = i > 0 and rsi[i] > rsi[i-1] if not np.isnan(rsi[i-1]) else False
        rsi_momentum_short = i > 0 and rsi[i] < rsi[i-1] if not np.isnan(rsi[i-1]) else False
        
        # === ENTRY LOGIC (LOOSE CONDITIONS) ===
        desired_signal = 0.0
        
        # LONG: HTF bull + RSI not overbought + optional local/macro confirmation
        if htf_bull and rsi_ok_long:
            # Strong signal: all confirmations align
            if local_bull and macro_bull:
                if is_choppy:
                    desired_signal = SIZE_BASE  # Reduce in chop
                else:
                    desired_signal = SIZE_STRONG
            # Base signal: HTF bull + RSI ok (minimum requirements)
            elif rsi_momentum_long:
                if is_choppy:
                    desired_signal = SIZE_BASE * 0.5
                else:
                    desired_signal = SIZE_BASE
        
        # SHORT: HTF bear + RSI not oversold + optional local/macro confirmation
        elif htf_bear and rsi_ok_short:
            # Strong signal: all confirmations align
            if local_bear and macro_bear:
                if is_choppy:
                    desired_signal = -SIZE_BASE  # Reduce in chop
                else:
                    desired_signal = -SIZE_STRONG
            # Base signal: HTF bear + RSI ok (minimum requirements)
            elif rsi_momentum_short:
                if is_choppy:
                    desired_signal = -SIZE_BASE * 0.5
                else:
                    desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            if low[i] < stop_price:
                stoploss_triggered = True
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            if high[i] > stop_price:
                stoploss_triggered = True
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
        
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
        elif abs(desired_signal) >= SIZE_BASE * 0.4:
            final_signal = np.sign(desired_signal) * SIZE_BASE * 0.5
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
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
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals
```

## Last Updated
2026-03-24 17:52
