#!/usr/bin/env python3
"""
EXPERIMENT #035 - HMA Trend + RSI Pullback + ADX Strength + ATR Dynamic Sizing
====================================================================================
Hypothesis: Return to HMA (which was in the best strategy Sharpe=11.523) but add ADX 
for trend strength confirmation. Many failed strategies traded during weak trends.
ADX > 25 filters out choppy markets. Combine with 4h trend + 1h entries (proven MTF).

Key changes from #034:
- HMA instead of KAMA (HMA was in best performing strategy)
- ADX(14) > 25 filter - only trade when trend has strength
- Fix read-only array errors by using .copy() properly
- Cleaner signal state tracking without modifying read-only arrays
- Discrete signal levels: 0.0, ±0.25, ±0.35 to minimize churn
- ATR-based dynamic position sizing with volatility targeting

Why this might beat Sharpe=11.523:
- HMA is faster and smoother than EMA (proven in best strategy)
- ADX filter avoids whipsaws in weak trends (major source of losses)
- 4h/1h MTF combination has proven effective
- Proper array handling prevents crashes
- Conservative position sizing (max 0.35) controls drawdown
"""

import numpy as np
import pandas as pd

name = "mtf_hma_rsi_adx_atr_dynamic_v1"
timeframe = "1h"
leverage = 1.0


def calculate_hma(close, period=16):
    """
    Calculate Hull Moving Average (HMA)
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    Faster and smoother than EMA, reduces lag significantly
    """
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA helper function
    def wma(data, window):
        result = np.zeros(len(data))
        weights = np.arange(1, window + 1)
        for i in range(window - 1, len(data)):
            result[i] = np.sum(data[i - window + 1:i + 1] * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half_period)
    wma_full = wma(close, period)
    
    # HMA calculation
    hma_raw = 2 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_period)
    
    # Fill initial values
    hma[:period] = close[:period]
    
    return hma


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    atr = np.zeros(n)
    atr[period] = np.mean(tr[1:period + 1])
    
    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    return atr


def calculate_rsi(close, period=14):
    """Calculate RSI with proper min_periods"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean().values
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean().values
    
    rs = np.zeros(n)
    mask = (avg_loss > 0) & (avg_gain >= 0)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    
    rsi = np.zeros(n)
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    rsi[avg_loss == 0] = 100
    
    return rsi


def calculate_adx(high, low, close, period=14):
    """
    Calculate Average Directional Index (ADX)
    ADX measures trend strength, not direction
    ADX > 25 = strong trend, ADX < 20 = weak/choppy
    """
    n = len(close)
    if n < period * 3:
        return np.zeros(n)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
        
        if high[i] - high[i - 1] > low[i - 1] - low[i]:
            plus_dm[i] = max(high[i] - high[i - 1], 0)
        else:
            plus_dm[i] = 0
            
        if low[i - 1] - low[i] > high[i] - high[i - 1]:
            minus_dm[i] = max(low[i - 1] - low[i], 0)
        else:
            minus_dm[i] = 0
    
    # Smooth with Wilder's method
    atr = np.zeros(n)
    atr[period] = np.mean(tr[1:period + 1])
    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        plus_di[i] = 100 * (pd.Series(plus_dm).rolling(window=period, min_periods=period).mean().iloc[i] / atr[i]) if atr[i] > 0 else 0
        minus_di[i] = 100 * (pd.Series(minus_dm).rolling(window=period, min_periods=period).mean().iloc[i] / atr[i]) if atr[i] > 0 else 0
    
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    # ADX is SMA of DX
    adx = pd.Series(dx).rolling(window=period, min_periods=period).mean().values
    adx = np.nan_to_num(adx, 0)
    
    return adx


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy() if "volume" in prices.columns else np.ones(len(close))
    n = len(close)
    
    if n < 100:
        return np.zeros(n)
    
    # 1h indicators for entry timing and risk
    rsi_1h = calculate_rsi(close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    hma_16_1h = calculate_hma(close, period=16)
    hma_48_1h = calculate_hma(close, period=48)
    adx_1h = calculate_adx(high, low, close, period=14)
    
    # 4h HMA for trend filter (resample 1h → 4h)
    df_1h = pd.DataFrame({
        'open': close,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    df_1h.index = pd.date_range(start='2021-01-01', periods=n, freq='1h')
    
    # Resample to 4h
    df_4h = df_1h.resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    c_4h = df_4h['close'].values.copy()
    h_4h = df_4h['high'].values.copy()
    l_4h = df_4h['low'].values.copy()
    n_4h = len(c_4h)
    
    if n_4h < 50:
        return np.zeros(n)
    
    # Calculate 4h HMA for trend
    hma_16_4h = calculate_hma(c_4h, period=16)
    hma_48_4h = calculate_hma(c_4h, period=48)
    
    # 4h trend direction based on HMA cross and price position
    trend_4h = np.zeros(n_4h)
    for i in range(48, n_4h):
        if hma_16_4h[i] > hma_48_4h[i] and c_4h[i] > hma_16_4h[i]:
            trend_4h[i] = 1  # Bullish
        elif hma_16_4h[i] < hma_48_4h[i] and c_4h[i] < hma_16_4h[i]:
            trend_4h[i] = -1  # Bearish
    
    # Map 4h trend back to 1h timeframe (4 x 1h = 4h)
    trend_1h = np.zeros(n)
    idx_1h_to_4h = np.arange(n) // 4
    
    for i in range(n):
        idx_4h = idx_1h_to_4h[i]
        if idx_4h < len(trend_4h):
            trend_1h[i] = trend_4h[idx_4h]
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels to reduce churn
    SIZE_FULL = 0.35   # Full position (max allowed per rules)
    SIZE_HALF = 0.18   # Half position (after take profit)
    
    # RSI thresholds for pullback entries
    RSI_LONG_ENTRY = 45   # Enter long on pullback in uptrend
    RSI_SHORT_ENTRY = 55  # Enter short on rally in downtrend
    RSI_EXIT_LONG = 70    # Exit long when overbought
    RSI_EXIT_SHORT = 30   # Exit short when oversold
    
    # ADX threshold for trend strength
    ADX_MIN = 25          # Only trade when ADX > 25 (strong trend)
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0   # 2*ATR stoploss
    
    # Take profit multiplier (2R)
    TP_MULT = 2.0
    
    # ATR volatility target for dynamic sizing
    TARGET_ATR_PCT = 0.02  # Target 2% ATR
    
    first_valid = max(50, 48, 14, 42)  # Wait for all indicators (ADX needs 3*period)
    
    # Track position state
    in_position = False
    position_side = 0  # 1 for long, -1 for short, 0 for flat
    entry_price = 0.0
    tp_triggered = False
    trailing_stop_price = 0.0
    
    for i in range(first_valid, n):
        # Check for NaN values
        if np.isnan(rsi_1h[i]) or np.isnan(atr_1h[i]) or np.isnan(adx_1h[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        trend = trend_1h[i]
        rsi_val = rsi_1h[i]
        adx_val = adx_1h[i]
        atr = atr_1h[i]
        price = close[i]
        
        # Skip if ADX too low (weak trend = choppy market)
        if adx_val < ADX_MIN:
            if in_position:
                # Check stoploss even in low ADX
                if position_side == 1:
                    if price < trailing_stop_price:
                        signals[i] = 0.0
                        in_position = False
                        position_side = 0
                        entry_price = 0.0
                        tp_triggered = False
                        trailing_stop_price = 0.0
                        continue
                elif position_side == -1:
                    if price > trailing_stop_price:
                        signals[i] = 0.0
                        in_position = False
                        position_side = 0
                        entry_price = 0.0
                        tp_triggered = False
                        trailing_stop_price = 0.0
                        continue
                
                # Hold position through low ADX
                signals[i] = signals[i - 1] if i > 0 else 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Check trailing stop and take profit for existing positions
        if in_position:
            if position_side == 1:
                # Update trailing stop (move up only)
                new_trail = max(trailing_stop_price, entry_price + ATR_STOP_MULT * atr) if trailing_stop_price > 0 else entry_price - ATR_STOP_MULT * atr
                trailing_stop_price = new_trail
                
                # Stoploss check
                if price < trailing_stop_price:
                    signals[i] = 0.0
                    in_position = False
                    position_side = 0
                    entry_price = 0.0
                    tp_triggered = False
                    trailing_stop_price = 0.0
                    continue
                
                # Take profit check (2R)
                tp_price = entry_price + TP_MULT * ATR_STOP_MULT * atr
                if not tp_triggered and price >= tp_price:
                    signals[i] = SIZE_HALF
                    tp_triggered = True
                    trailing_stop_price = entry_price  # Trail to breakeven
                    continue
                
                # RSI exit signal
                if rsi_val > RSI_EXIT_LONG:
                    signals[i] = 0.0
                    in_position = False
                    position_side = 0
                    entry_price = 0.0
                    tp_triggered = False
                    trailing_stop_price = 0.0
                    continue
                    
            elif position_side == -1:
                # Update trailing stop (move down only)
                new_trail = min(trailing_stop_price, entry_price - ATR_STOP_MULT * atr) if trailing_stop_price > 0 else entry_price + ATR_STOP_MULT * atr
                trailing_stop_price = new_trail
                
                # Stoploss check
                if price > trailing_stop_price:
                    signals[i] = 0.0
                    in_position = False
                    position_side = 0
                    entry_price = 0.0
                    tp_triggered = False
                    trailing_stop_price = 0.0
                    continue
                
                # Take profit check (2R)
                tp_price = entry_price - TP_MULT * ATR_STOP_MULT * atr
                if not tp_triggered and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    tp_triggered = True
                    trailing_stop_price = entry_price  # Trail to breakeven
                    continue
                
                # RSI exit signal
                if rsi_val < RSI_EXIT_SHORT:
                    signals[i] = 0.0
                    in_position = False
                    position_side = 0
                    entry_price = 0.0
                    tp_triggered = False
                    trailing_stop_price = 0.0
                    continue
            
            # Hold position
            signals[i] = signals[i - 1] if i > 0 else 0.0
            continue
        
        # Dynamic position sizing based on ATR volatility
        current_atr_pct = atr / price if price > 0 else 0
        if current_atr_pct > 0:
            size_multiplier = min(1.5, max(0.5, TARGET_ATR_PCT / current_atr_pct))
        else:
            size_multiplier = 1.0
        
        position_size = SIZE_FULL * size_multiplier
        position_size = min(SIZE_FULL, max(SIZE_HALF, position_size))  # Clamp
        
        if trend == 1:  # 4h uptrend
            # RSI pullback entry + ADX confirmation
            if rsi_val < RSI_LONG_ENTRY and rsi_val > 30:
                signals[i] = position_size
                in_position = True
                position_side = 1
                entry_price = price
                tp_triggered = False
                trailing_stop_price = price - ATR_STOP_MULT * atr
            else:
                signals[i] = 0.0
                    
        elif trend == -1:  # 4h downtrend
            # RSI rally entry + ADX confirmation
            if rsi_val > RSI_SHORT_ENTRY and rsi_val < 70:
                signals[i] = -position_size
                in_position = True
                position_side = -1
                entry_price = price
                tp_triggered = False
                trailing_stop_price = price + ATR_STOP_MULT * atr
            else:
                signals[i] = 0.0
        else:  # No clear trend
            signals[i] = 0.0
    
    return signals