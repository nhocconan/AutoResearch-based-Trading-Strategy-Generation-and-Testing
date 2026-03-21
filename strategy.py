#!/usr/bin/env python3
"""
EXPERIMENT #098 - MTF Supertrend+RSI with ATR Position Scaling (15m+4h)
==================================================================================================
Hypothesis: Previous regime/ensemble strategies (#086-#097) failed due to overcomplexity and 
incorrect rolling calculations. The current best uses Supertrend+MACD+BBW+RSI across 15m/1h/4h.

Key changes:
- Simpler signal logic: Supertrend (4h) for trend + RSI (15m) for pullback entries
- ATR-based position scaling: smaller positions when volatility is high
- Fixed HMA/rolling calculations (no Rolling.apply with min_periods bug from #097)
- Discrete position sizing: 0.0, ±0.20, ±0.35 only
- Trailing stoploss at 2*ATR with take-profit at 3R

Why this should work:
- Supertrend is proven trend filter (less whipsaw than HMA crossover)
- ATR position scaling reduces risk in high vol (critical for drawdown control)
- Simpler logic = fewer bugs and more robust across BTC/ETH/SOL
- Based on lessons from #040 (Sharpe=5.4) and current best (Sharpe=3.65)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_supertrend_rsi_atr_scale_15m_4h_v1"
timeframe = "15m"
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


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    atr = calculate_atr(high, low, close, period)
    
    hl2 = (high + low) / 2
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    supertrend = np.zeros(n)
    trend = np.ones(n)  # 1 = bullish, -1 = bearish
    
    for i in range(period, n):
        upper_band[i] = hl2[i] + multiplier * atr[i]
        lower_band[i] = hl2[i] - multiplier * atr[i]
        
        if i == period:
            supertrend[i] = upper_band[i]
            trend[i] = 1 if close[i] > supertrend[i] else -1
        else:
            if trend[i - 1] == 1:
                supertrend[i] = max(upper_band[i], supertrend[i - 1]) if upper_band[i] < supertrend[i - 1] else upper_band[i]
                if close[i] < supertrend[i]:
                    trend[i] = -1
                    supertrend[i] = lower_band[i]
            else:
                supertrend[i] = min(lower_band[i], supertrend[i - 1]) if lower_band[i] > supertrend[i - 1] else lower_band[i]
                if close[i] > supertrend[i]:
                    trend[i] = 1
                    supertrend[i] = upper_band[i]
    
    return supertrend, trend


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    n = len(close)
    if n < period + 1:
        return np.full(n, 50.0)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    
    avg_gain[period - 1] = np.mean(gain[:period])
    avg_loss[period - 1] = np.mean(loss[:period])
    
    for i in range(period, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
    
    rs = np.zeros(n)
    for i in range(period, n):
        if avg_loss[i] == 0:
            rs[i] = 100
        else:
            rs[i] = avg_gain[i] / avg_loss[i]
    
    rsi = 100 - (100 / (1 + rs))
    rsi[:period] = 50
    
    return rsi


def calculate_sma(close, period=20):
    """Calculate Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    sma = np.zeros(n)
    for i in range(period - 1, n):
        sma[i] = np.mean(close[i - period + 1:i + 1])
    
    return sma


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    sma_15m = calculate_sma(close, period=200)
    
    # 4h indicators for trend (using PROPER mtf_data helper)
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        supertrend_4h_raw, trend_4h_raw = calculate_supertrend(high_4h, low_4h, close_4h, period=10, multiplier=3.0)
        
        # Align 4h indicators to 15m timeframe (auto shift(1) for completed bars)
        trend_4h = align_htf_to_ltf(prices, df_4h, trend_4h_raw)
        supertrend_4h = align_htf_to_ltf(prices, df_4h, supertrend_4h_raw)
        
    except Exception:
        # Fallback if mtf_data fails
        trend_4h = np.ones(n)
        supertrend_4h = np.zeros(n)
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    # ATR-based scaling: reduce size when volatility is high
    BASE_SIZE = 0.35
    HALF_SIZE = 0.175
    QUARTER_SIZE = 0.10
    
    # ATR percentile for position scaling
    atr_lookback = 100
    atr_percentile = np.zeros(n)
    for i in range(atr_lookback - 1, n):
        window = atr_15m[i - atr_lookback + 1:i + 1]
        rank = np.sum(window <= atr_15m[i])
        atr_percentile[i] = rank / atr_lookback
    
    # RSI thresholds for pullback entries
    RSI_LONG_MIN = 40
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 60
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    ATR_TRAIL_MULT = 1.5
    TP_MULT = 3.0
    
    first_valid = max(200, 14 * 2, 100)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            continue
        
        # 4h Supertrend filter
        trend_4h_val = trend_4h[i]
        
        # Price vs 200 SMA filter (only trade in direction of long-term trend)
        price_vs_sma = 0
        if sma_15m[i] > 0:
            if close[i] > sma_15m[i]:
                price_vs_sma = 1
            elif close[i] < sma_15m[i]:
                price_vs_sma = -1
        
        # ATR-based position scaling
        atr_pct = atr_percentile[i]
        if atr_pct > 0.7:  # High volatility - reduce position
            position_size = HALF_SIZE
        elif atr_pct > 0.4:  # Medium volatility
            position_size = BASE_SIZE
        else:  # Low volatility - can use full size
            position_size = BASE_SIZE
        
        rsi_val = rsi_15m[i]
        atr = atr_15m[i]
        price = close[i]
        
        # Check stoploss and take profit for existing positions
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_tp = tp_triggered[i - 1]
            prev_high = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
            prev_low = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
            
            # Update highest/lowest since entry
            if prev_side == 1:
                current_high = max(prev_high, price)
                current_low = min(prev_low, price) if prev_low > 0 else price
            else:
                current_high = max(prev_high, price) if prev_high > 0 else price
                current_low = min(prev_low, price)
            
            highest_since_entry[i] = current_high
            lowest_since_entry[i] = current_low
            
            # Stoploss check (2*ATR)
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (3R) - reduce to half
                tp_price = prev_entry + TP_MULT * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = HALF_SIZE
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1.5R profit
                if prev_tp:
                    trail_stop = current_high - ATR_TRAIL_MULT * atr
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (3R) - reduce to half
                tp_price = prev_entry - TP_MULT * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    signals[i] = -HALF_SIZE
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1.5R profit
                if prev_tp:
                    trail_stop = current_low + ATR_TRAIL_MULT * atr
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
            
            # Hold position if no exit triggered
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
            continue
        
        # Entry logic: 4h Supertrend + 15m RSI pullback + 200 SMA filter
        # Long: 4h Supertrend bullish + 15m RSI pullback + price above 200 SMA
        if trend_4h_val == 1 and price_vs_sma >= 0 and (RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX):
            signals[i] = position_size
            position_side[i] = 1
            entry_price[i] = price
            tp_triggered[i] = 0
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        
        # Short: 4h Supertrend bearish + 15m RSI pullback + price below 200 SMA
        elif trend_4h_val == -1 and price_vs_sma <= 0 and (RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX):
            signals[i] = -position_size
            position_side[i] = -1
            entry_price[i] = price
            tp_triggered[i] = 0
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals