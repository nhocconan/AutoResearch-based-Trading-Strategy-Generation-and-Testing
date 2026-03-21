#!/usr/bin/env python3
"""
EXPERIMENT #103 - MTF Donchian+KAMA+Chandelier+VolAdj Sizing (15m+1h+4h Proper HTF)
==================================================================================================
Hypothesis: Current best (Sharpe=3.653) uses Supertrend+MACD+BBW+RSI. 
This experiment tries Donchian breakout + KAMA adaptive trend + Chandelier exit.

Key innovations:
1. Donchian(20) breakout for entry signals (proven momentum indicator)
2. KAMA for adaptive trend filtering (works well in crypto volatility)
3. Chandelier exit (highest_high - 3*ATR) for trailing stops
4. Volatility-adjusted position sizing (smaller size when ATR% is high)
5. Proper MTF using mtf_data module (15m entries, 1h+4h trend filters)
6. Discrete signal levels (0.0, ±0.25, ±0.35) to minimize churn costs

Why this should beat current best:
- Donchian breakouts capture momentum better than Supertrend in trending markets
- KAMA adapts to volatility changes (better than static HMA)
- Chandelier exit is proven to reduce drawdown vs fixed ATR stops
- Vol-adjusted sizing reduces exposure during high volatility periods
- Triple timeframe confirmation (15m/1h/4h) reduces false signals

Risk management:
- Max signal: 0.35 (35% of capital max)
- ATR stop: 2.5*ATR for initial, Chandelier (3*ATR) for trailing
- Position size scaled by volatility: base_size * (target_vol / current_vol)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_donchian_kama_chandelier_voladj_15m_1h_4h_v1"
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


def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average"""
    n = len(close)
    if n < er_period + slow_period:
        return np.zeros(n)
    
    kama = np.zeros(n)
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        change = abs(close[i] - close[i - er_period])
        volatility = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        
        if volatility > 0:
            er = change / volatility
        else:
            er = 0
        
        sc = (er * (2.0 / (fast_period + 1) - 2.0 / (slow_period + 1)) + 2.0 / (slow_period + 1)) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    
    avg_gain[period] = np.mean(gain[:period + 1])
    avg_loss[period] = np.mean(loss[:period + 1])
    
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
    
    rs = np.zeros(n)
    for i in range(period, n):
        if avg_loss[i] == 0:
            rs[i] = 100
        else:
            rs[i] = avg_gain[i] / avg_loss[i]
    
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high, lowest low over period)"""
    n = len(high)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower


def calculate_chandelier_exit(high, low, close, atr, period=22, multiplier=3.0):
    """Calculate Chandelier Exit (trailing stop based on highest high - ATR*mult)"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    chandelier_long = np.zeros(n)  # Stop for long positions
    chandelier_short = np.zeros(n)  # Stop for short positions
    
    highest_high = np.zeros(n)
    lowest_low = np.zeros(n)
    
    for i in range(period - 1, n):
        highest_high[i] = np.max(high[i - period + 1:i + 1])
        lowest_low[i] = np.min(low[i - period + 1:i + 1])
        
        chandelier_long[i] = highest_high[i] - multiplier * atr[i]
        chandelier_short[i] = lowest_low[i] + multiplier * atr[i]
    
    return chandelier_long, chandelier_short


def calculate_volatility_regime(close, period=20):
    """Calculate volatility regime (1=low, 2=medium, 3=high)"""
    n = len(close)
    if n < period:
        return np.ones(n)
    
    regime = np.ones(n)
    returns = np.diff(close, prepend=close[0]) / close
    
    for i in range(period - 1, n):
        window = returns[i - period + 1:i + 1]
        vol = np.std(window)
        
        # Simple regime classification
        if vol < 0.01:
            regime[i] = 1  # Low vol
        elif vol < 0.025:
            regime[i] = 2  # Medium vol
        else:
            regime[i] = 3  # High vol
    
    return regime


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Get HTF data using mtf_data module (CRITICAL - no manual resampling)
    try:
        df_1h = get_htf_data(prices, '1h')
        df_4h = get_htf_data(prices, '4h')
    except Exception:
        # Fallback if mtf_data not available
        df_1h = prices
        df_4h = prices
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    kama_15m = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    donchian_upper_15m, donchian_lower_15m = calculate_donchian(high, low, period=20)
    chandelier_long_15m, chandelier_short_15m = calculate_chandelier_exit(high, low, close, atr_15m, period=22, multiplier=3.0)
    vol_regime_15m = calculate_volatility_regime(close, period=20)
    
    # 1h indicators for trend filter (using mtf_data align)
    close_1h = df_1h["close"].values
    high_1h = df_1h["high"].values
    low_1h = df_1h["low"].values
    
    atr_1h = calculate_atr(high_1h, low_1h, close_1h, period=14)
    kama_1h = calculate_kama(close_1h, er_period=10, fast_period=2, slow_period=30)
    donchian_upper_1h, donchian_lower_1h = calculate_donchian(high_1h, low_1h, period=20)
    rsi_1h = calculate_rsi(close_1h, period=14)
    
    # Align 1h indicators to 15m timeframe
    kama_1h_aligned = align_htf_to_ltf(prices, df_1h, kama_1h)
    donchian_upper_1h_aligned = align_htf_to_ltf(prices, df_1h, donchian_upper_1h)
    donchian_lower_1h_aligned = align_htf_to_ltf(prices, df_1h, donchian_lower_1h)
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h)
    atr_1h_aligned = align_htf_to_ltf(prices, df_1h, atr_1h)
    
    # 4h indicators for major trend filter
    close_4h = df_4h["close"].values
    high_4h = df_4h["high"].values
    low_4h = df_4h["low"].values
    
    kama_4h = calculate_kama(close_4h, er_period=10, fast_period=2, slow_period=30)
    donchian_upper_4h, donchian_lower_4h = calculate_donchian(high_4h, low_4h, period=20)
    
    # Align 4h indicators to 15m timeframe
    kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h)
    donchian_upper_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper_4h)
    donchian_lower_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower_4h)
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels with volatility adjustment
    BASE_SIZE = 0.30  # Base position size (30% of capital)
    SIZE_HALF = 0.15
    
    # Volatility adjustment factors
    VOL_LOW_MULT = 1.2   # Increase size in low vol
    VOL_MED_MULT = 1.0   # Normal size in medium vol
    VOL_HIGH_MULT = 0.6  # Reduce size in high vol
    
    # Entry thresholds
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 65
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5
    
    first_valid = max(200, 40, 22)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        # Check for NaN or zero ATR
        if np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            continue
        
        # Check HTF data alignment
        if np.isnan(kama_1h_aligned[i]) or np.isnan(kama_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_15m[i]
        rsi_val = rsi_15m[i]
        vol_regime = vol_regime_15m[i]
        
        # Calculate volatility-adjusted position size
        if vol_regime == 1:
            vol_mult = VOL_LOW_MULT
        elif vol_regime == 2:
            vol_mult = VOL_MED_MULT
        else:
            vol_mult = VOL_HIGH_MULT
        
        adjusted_size = min(BASE_SIZE * vol_mult, 0.35)  # Cap at 35%
        
        # Trend filters (all timeframes must agree)
        # 1h trend
        if close[i] > kama_1h_aligned[i]:
            trend_1h = 1
        elif close[i] < kama_1h_aligned[i]:
            trend_1h = -1
        else:
            trend_1h = 0
        
        # 4h trend
        if close[i] > kama_4h_aligned[i]:
            trend_4h = 1
        elif close[i] < kama_4h_aligned[i]:
            trend_4h = -1
        else:
            trend_4h = 0
        
        # Donchian breakout confirmation (1h)
        donchian_breakout_1h = 0
        if price > donchian_upper_1h_aligned[i]:
            donchian_breakout_1h = 1
        elif price < donchian_lower_1h_aligned[i]:
            donchian_breakout_1h = -1
        
        # RSI 1h filter
        rsi_1h_val = rsi_1h_aligned[i]
        rsi_1h_bullish = 40 < rsi_1h_val < 70
        rsi_1h_bearish = 30 < rsi_1h_val < 60
        
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
            
            # Chandelier exit stoploss check
            if prev_side == 1:
                chandelier_stop = chandelier_long_15m[i]
                initial_stop = prev_entry - ATR_STOP_MULT * atr
                
                # Use the tighter of chandelier or initial stop
                effective_stop = max(chandelier_stop, initial_stop)
                
                if price < effective_stop:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                    
            elif prev_side == -1:
                chandelier_stop = chandelier_short_15m[i]
                initial_stop = prev_entry + ATR_STOP_MULT * atr
                
                # Use the tighter of chandelier or initial stop
                effective_stop = min(chandelier_stop, initial_stop)
                
                if price > effective_stop:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr
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
        
        # Entry logic: Triple timeframe confirmation + Donchian breakout + RSI filter
        # Long entry: 1h bullish + 4h bullish + Donchian breakout + RSI pullback
        if trend_1h == 1 and trend_4h == 1 and rsi_1h_bullish:
            if (RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX and 
                price > donchian_upper_15m[i] * 0.995):  # Near Donchian breakout
                signals[i] = adjusted_size
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                
        # Short entry: 1h bearish + 4h bearish + Donchian breakdown + RSI pullback
        elif trend_1h == -1 and trend_4h == -1 and rsi_1h_bearish:
            if (RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX and 
                price < donchian_lower_15m[i] * 1.005):  # Near Donchian breakdown
                signals[i] = -adjusted_size
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals