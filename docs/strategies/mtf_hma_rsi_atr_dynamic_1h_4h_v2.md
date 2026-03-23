# Strategy: mtf_hma_rsi_atr_dynamic_1h_4h_v2

## Status
ACTIVE - Sharpe=0.665 | Return=+123.5% | DD=-21.8%

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | 0.190 | +30.9% | -22.0% | 463 |
| ETHUSDT | 0.450 | +58.5% | -23.9% | 504 |
| SOLUSDT | 1.354 | +281.0% | -19.5% | 508 |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | -0.867 | -5.7% | -13.8% | 137 |
| ETHUSDT | -0.286 | -1.9% | -21.0% | 151 |
| SOLUSDT | -0.889 | -16.8% | -27.5% | 162 |

## Code
```python
#!/usr/bin/env python3
"""
EXPERIMENT #036 - MTF HMA+RSI+ATR Dynamic Sizing (1h+4h v2)
==================================================================================================
Hypothesis: Simplify the MTF approach by using 1h base + 4h trend (removing 15m complexity).
The current best (30m+4h) has Sharpe=1.153. By moving to 1h base:
- Fewer false signals (1h candles are more reliable than 30m)
- Lower transaction costs (fewer trades but higher quality)
- Cleaner trend following with HMA(21/48) on 4h
- RSI pullback entries on 1h with wider thresholds (30-70 instead of 35-55)
- ATR-based dynamic position sizing (reduce size when volatility is high)

Why this should beat current best:
- 1h timeframe proven in experiment #027 (Sharpe=0.102, +60% return)
- Simpler logic = fewer whipsaws = better Sharpe
- Dynamic sizing reduces exposure during high volatility periods
- Remove BBW filter (was killing trade count in #031)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_hma_rsi_atr_dynamic_1h_4h_v2"
timeframe = "1h"
leverage = 1.0


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    atr = np.zeros(n)
    atr[period - 1] = np.mean(tr[1:period])
    
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    return atr


def calculate_hma(close, period=21):
    """Calculate Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half_period, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, adjust=False).mean().values
    
    hma = pd.Series(2 * wma1 - wma2).ewm(span=sqrt_period, adjust=False).mean().values
    
    return hma


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False).mean().values
    
    rs = np.zeros(n)
    for i in range(n):
        if avg_loss[i] == 0:
            rs[i] = 100
        else:
            rs[i] = avg_gain[i] / avg_loss[i]
    
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average"""
    n = len(close)
    if n < er_period + slow_period:
        return np.zeros(n)
    
    # Efficiency Ratio
    change = np.zeros(n)
    volatility = np.zeros(n)
    
    for i in range(er_period, n):
        change[i] = abs(close[i] - close[i - er_period])
        volatility[i] = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
    
    er = np.zeros(n)
    for i in range(er_period, n):
        if volatility[i] > 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 0
    
    # Smoothing constant
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    
    kama = np.zeros(n)
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 1h indicators for entry timing
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    hma_fast_1h = calculate_hma(close, period=21)
    hma_slow_1h = calculate_hma(close, period=48)
    
    # Get 4h data using mtf_data helper (MUST use this for proper alignment)
    try:
        df_4h = get_htf_data(prices, '4h')
        c_4h = df_4h['close'].values
        h_4h = df_4h['high'].values
        l_4h = df_4h['low'].values
        
        # 4h HMA for trend direction
        hma_fast_4h = calculate_hma(c_4h, period=21)
        hma_slow_4h = calculate_hma(c_4h, period=48)
        kama_4h = calculate_kama(c_4h, er_period=10)
        rsi_4h = calculate_rsi(c_4h, period=14)
        
        # Align 4h indicators to 1h timeframe (auto shift for completed bars)
        hma_fast_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_fast_4h)
        hma_slow_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_slow_4h)
        kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h)
        rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    except Exception:
        # Fallback if mtf_data fails
        hma_fast_4h_aligned = np.zeros(n)
        hma_slow_4h_aligned = np.zeros(n)
        kama_4h_aligned = np.zeros(n)
        rsi_4h_aligned = np.zeros(n) + 50
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels with ATR-based dynamic adjustment
    BASE_SIZE = 0.30
    SIZE_HALF = 0.15
    TARGET_ATR_PCT = 0.02  # Target 2% ATR as % of price
    
    # RSI thresholds for pullback entries (wider than 15m strategy)
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 65
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5  # Slightly wider stops for 1h timeframe
    
    # Minimum volatility filter (avoid extremely low vol)
    MIN_ATR_PCT = 0.005
    
    first_valid = max(200, 48, 30)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    entry_atr = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_1h[i]) or np.isnan(rsi_1h[i]) or atr_1h[i] == 0:
            signals[i] = 0.0
            continue
        
        # Calculate ATR as % of price for dynamic sizing
        atr_pct = atr_1h[i] / close[i] if close[i] > 0 else 0
        
        # Skip if volatility is extremely low (choppy market)
        if atr_pct < MIN_ATR_PCT:
            signals[i] = 0.0
            if i > 0:
                position_side[i] = 0
            continue
        
        # Dynamic position sizing based on current volatility
        if atr_pct > 0:
            vol_adjustment = min(TARGET_ATR_PCT / atr_pct, 1.5)  # Cap at 1.5x
            size_full = min(BASE_SIZE * vol_adjustment, 0.40)  # Max 40%
            size_half = size_full / 2
        else:
            size_full = BASE_SIZE
            size_half = SIZE_HALF
        
        # Get aligned MTF values
        hma_fast_4h_val = hma_fast_4h_aligned[i] if i < len(hma_fast_4h_aligned) else 0
        hma_slow_4h_val = hma_slow_4h_aligned[i] if i < len(hma_slow_4h_aligned) else 0
        kama_4h_val = kama_4h_aligned[i] if i < len(kama_4h_aligned) else 0
        rsi_4h_val = rsi_4h_aligned[i] if i < len(rsi_4h_aligned) else 50
        
        # 4h trend filter: HMA fast > HMA slow = bullish, HMA fast < HMA slow = bearish
        # Also require price > KAMA for bullish, price < KAMA for bearish
        trend_4h = 0
        if hma_fast_4h_val > 0 and hma_slow_4h_val > 0:
            if hma_fast_4h_val > hma_slow_4h_val and c_4h is not None and len(c_4h) > 0:
                idx_4h = min(i // 4, len(c_4h) - 1)  # 4 x 1h = 4h
                if idx_4h >= 0 and idx_4h < len(c_4h) and c_4h[idx_4h] > kama_4h_val:
                    trend_4h = 1
            elif hma_fast_4h_val < hma_slow_4h_val and c_4h is not None and len(c_4h) > 0:
                idx_4h = min(i // 4, len(c_4h) - 1)
                if idx_4h >= 0 and idx_4h < len(c_4h) and c_4h[idx_4h] < kama_4h_val:
                    trend_4h = -1
        
        # 4h RSI filter (avoid extreme overbought/oversold for counter-trend)
        rsi_4h_ok = True
        if trend_4h == 1 and rsi_4h_val > 75:
            rsi_4h_ok = False  # Too overbought on 4h
        elif trend_4h == -1 and rsi_4h_val < 25:
            rsi_4h_ok = False  # Too oversold on 4h
        
        # Check stoploss and take profit for existing positions
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_tp = tp_triggered[i - 1]
            prev_high = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
            prev_low = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
            prev_atr = entry_atr[i - 1] if entry_atr[i - 1] > 0 else atr_1h[i - 1]
            
            price = close[i]
            
            # Update highest/lowest since entry
            if prev_side == 1:
                current_high = max(prev_high, price)
                current_low = min(prev_low, price)
            else:
                current_high = max(prev_high, price)
                current_low = min(prev_low, price)
            
            highest_since_entry[i] = current_high
            lowest_since_entry[i] = current_low
            
            # Stoploss check (2.5*ATR from entry)
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * prev_atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    entry_atr[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * prev_atr
                if not prev_tp and price >= tp_price:
                    signals[i] = size_half
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    highest_since_entry[i] = current_high
                    lowest_since_entry[i] = current_low
                    entry_atr[i] = prev_atr
                    continue
                
                # Trail stop at 1R profit after TP triggered
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * prev_atr
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        entry_atr[i] = 0
                        continue
            
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * prev_atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    entry_atr[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * prev_atr
                if not prev_tp and price <= tp_price:
                    signals[i] = -size_half
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    highest_since_entry[i] = current_high
                    lowest_since_entry[i] = current_low
                    entry_atr[i] = prev_atr
                    continue
                
                # Trail stop at 1R profit after TP triggered
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * prev_atr
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        entry_atr[i] = 0
                        continue
            
            # Check if trend changed - exit position
            if trend_4h != prev_side and trend_4h != 0:
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
                tp_triggered[i] = 0
                highest_since_entry[i] = 0
                lowest_since_entry[i] = 0
                entry_atr[i] = 0
                continue
            
            # Hold position if no exit triggered
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
            entry_atr[i] = entry_atr[i - 1]
            continue
        
        # Entry logic: 4h trend + 1h RSI pullback
        price = close[i]
        
        # 1h HMA alignment check (fast > slow for long, fast < slow for short)
        hma_aligned_1h = 0
        if hma_fast_1h[i] > hma_slow_1h[i]:
            hma_aligned_1h = 1
        elif hma_fast_1h[i] < hma_slow_1h[i]:
            hma_aligned_1h = -1
        
        if trend_4h == 1 and rsi_4h_ok and hma_aligned_1h == 1:  # Bullish trend
            # RSI pullback on 1h (not overbought)
            if RSI_LONG_MIN <= rsi_1h[i] <= RSI_LONG_MAX:
                signals[i] = size_full
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                entry_atr[i] = atr_1h[i]
                
        elif trend_4h == -1 and rsi_4h_ok and hma_aligned_1h == -1:  # Bearish trend
            # RSI pullback on 1h (not oversold)
            if RSI_SHORT_MIN <= rsi_1h[i] <= RSI_SHORT_MAX:
                signals[i] = -size_full
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                entry_atr[i] = atr_1h[i]
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals
```

## Last Updated
2026-03-21 16:59
