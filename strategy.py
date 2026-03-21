#!/usr/bin/env python3
"""
EXPERIMENT #038 - HMA Trend + RSI Pullback + ATR Trailing Stop + 1h Timeframe
====================================================================================
Hypothesis: Building on #021's success, shift from 15m to 1h timeframe for fewer whipsaws
and lower transaction costs. 1h provides better signal-to-noise ratio while maintaining
responsiveness. Keep 4h HMA for trend filter but use 1h RSI for entries.

Key improvements over #021:
- 1h timeframe instead of 15m - fewer false signals, lower fees
- Tighter RSI thresholds (40/60 vs 45/55) for cleaner entries
- ATR trailing stop that moves with price (not fixed at entry)
- Position sizing capped at 0.30 max (more conservative than 0.35)
- Add Bollinger Band width filter to avoid low-volatility chop

Why this might beat Sharpe=11.523:
- 1h has better risk/reward than 15m (less noise, similar capture)
- Fewer signal changes = lower transaction costs (0.10% per change)
- Trailing stop locks in more profit on strong trends
- BB width filter avoids dead zones
"""

import numpy as np
import pandas as pd

name = "mtf_hma_rsi_atr_trail_1h_v1"
timeframe = "1h"
leverage = 1.0


def calculate_hma(close, period=16):
    """Calculate Hull Moving Average - reduces lag vs EMA"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    if sqrt_period < 1:
        sqrt_period = 1
    if half < 1:
        half = 1
    
    # WMA helper function
    def wma(arr, w):
        result = np.zeros(len(arr))
        weights = np.arange(1, w + 1, dtype=np.float64)
        w_sum = np.sum(weights)
        for i in range(w - 1, len(arr)):
            result[i] = np.sum(arr[i - w + 1:i + 1] * weights) / w_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    hma_raw = 2 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_period)
    
    return hma


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    tr = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    atr = np.zeros(n)
    if n >= period:
        atr[period - 1] = np.mean(tr[1:period])
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    return atr


def calculate_rsi(close, period=14):
    """Calculate RSI with proper min_periods"""
    n = len(close)
    delta = np.zeros(n)
    delta[1:] = np.diff(close)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean().values
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean().values
    
    rs = np.zeros(n)
    mask = (avg_loss > 0) & (~np.isnan(avg_loss))
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    
    rsi = np.zeros(n)
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    
    return rsi


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands"""
    n = len(close)
    mean = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = mean + std_mult * std
    lower = mean - std_mult * std
    width = (upper - lower) / mean
    
    return upper, lower, width


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64) if "volume" in prices.columns else np.ones(len(close))
    n = len(close)
    
    # 1h indicators for entry timing and risk
    rsi_1h = calculate_rsi(close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    bb_upper, bb_lower, bb_width = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    hma_16_1h = calculate_hma(close, period=16)
    hma_48_1h = calculate_hma(close, period=48)
    
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
    
    c_4h = df_4h['close'].values.astype(np.float64)
    h_4h = df_4h['high'].values.astype(np.float64)
    l_4h = df_4h['low'].values.astype(np.float64)
    
    # Calculate 4h HMA for trend
    hma_16_4h = calculate_hma(c_4h, period=16)
    hma_48_4h = calculate_hma(c_4h, period=48)
    
    # 4h trend direction based on HMA cross and price position
    trend_4h = np.zeros(len(c_4h))
    for i in range(48, len(c_4h)):
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
    SIZE_FULL = 0.30   # Full position (reduced from 0.35)
    SIZE_HALF = 0.15   # Half position (after take profit)
    
    # RSI thresholds for pullback entries (tighter than 15m version)
    RSI_LONG_ENTRY = 40   # Enter long on pullback in uptrend
    RSI_SHORT_ENTRY = 60  # Enter short on rally in downtrend
    RSI_EXIT_LONG = 70    # Exit long when overbought
    RSI_EXIT_SHORT = 30   # Exit short when oversold
    
    # Bollinger Band width filter (avoid low volatility chop)
    BB_WIDTH_MIN = 0.02   # Minimum BB width % to trade
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    
    # Take profit multiplier (2R)
    TP_MULT = 2.0
    
    # ATR volatility target for dynamic sizing
    TARGET_ATR_PCT = 0.015  # Target 1.5% ATR (lower for 1h)
    
    first_valid = max(80, 48, 14, 20)  # Wait for all indicators
    
    # Track position state
    position_side = 0  # 1 for long, -1 for short, 0 for flat
    entry_price = 0.0
    tp_triggered = False
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    trailing_stop = 0.0
    
    for i in range(first_valid, n):
        if np.isnan(rsi_1h[i]) or np.isnan(atr_1h[i]) or np.isnan(bb_width[i]):
            signals[i] = 0.0
            continue
        
        trend = trend_1h[i]
        rsi_val = rsi_1h[i]
        bb_w = bb_width[i]
        atr = atr_1h[i]
        price = close[i]
        
        # BB width filter - avoid low volatility chop
        if bb_w < BB_WIDTH_MIN:
            if position_side != 0:
                # Close existing position
                signals[i] = 0.0
                position_side = 0
                entry_price = 0.0
                tp_triggered = False
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                trailing_stop = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # ATR filter - avoid trading when ATR is extremely high
        if atr > 0 and atr / price > 0.05:  # ATR > 5% of price = too volatile
            if position_side != 0:
                signals[i] = 0.0
                position_side = 0
                entry_price = 0.0
                tp_triggered = False
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                trailing_stop = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Check trailing stop and take profit for existing positions
        if position_side != 0:
            prev_side = position_side
            
            # Update highest/lowest since entry for trailing
            if prev_side == 1:
                highest_since_entry = max(highest_since_entry, price)
                lowest_since_entry = min(lowest_since_entry, price) if lowest_since_entry > 0 else price
                
                # Update trailing stop (move up only)
                new_trailing = highest_since_entry - ATR_STOP_MULT * atr
                if new_trailing > trailing_stop:
                    trailing_stop = new_trailing
                
                # Stoploss check
                if price < trailing_stop:
                    signals[i] = 0.0
                    position_side = 0
                    entry_price = 0.0
                    tp_triggered = False
                    highest_since_entry = 0.0
                    lowest_since_entry = 0.0
                    trailing_stop = 0.0
                    continue
                
                # Take profit check (2R)
                tp_price = entry_price + TP_MULT * ATR_STOP_MULT * atr
                if not tp_triggered and price >= tp_price:
                    # Reduce to half position
                    signals[i] = SIZE_HALF
                    position_side = 1
                    tp_triggered = True
                    # Trail stop to breakeven
                    trailing_stop = max(trailing_stop, entry_price)
                    continue
                
                # RSI exit signal
                if rsi_val > RSI_EXIT_LONG:
                    signals[i] = 0.0
                    position_side = 0
                    entry_price = 0.0
                    tp_triggered = False
                    highest_since_entry = 0.0
                    lowest_since_entry = 0.0
                    trailing_stop = 0.0
                    continue
                    
            elif prev_side == -1:
                lowest_since_entry = min(lowest_since_entry, price) if lowest_since_entry > 0 else price
                highest_since_entry = max(highest_since_entry, price)
                
                # Update trailing stop (move down only)
                new_trailing = lowest_since_entry + ATR_STOP_MULT * atr
                if new_trailing < trailing_stop or trailing_stop == 0:
                    trailing_stop = new_trailing
                
                # Stoploss check
                if price > trailing_stop:
                    signals[i] = 0.0
                    position_side = 0
                    entry_price = 0.0
                    tp_triggered = False
                    highest_since_entry = 0.0
                lowest_since_entry = 0.0
                    trailing_stop = 0.0
                    continue
                
                # Take profit check (2R)
                tp_price = entry_price - TP_MULT * ATR_STOP_MULT * atr
                if not tp_triggered and price <= tp_price:
                    # Reduce to half position
                    signals[i] = -SIZE_HALF
                    position_side = -1
                    tp_triggered = True
                    # Trail stop to breakeven
                    trailing_stop = min(trailing_stop, entry_price) if trailing_stop > 0 else entry_price
                    continue
                
                # RSI exit signal
                if rsi_val < RSI_EXIT_SHORT:
                    signals[i] = 0.0
                    position_side = 0
                    entry_price = 0.0
                    tp_triggered = False
                    highest_since_entry = 0.0
                    lowest_since_entry = 0.0
                    trailing_stop = 0.0
                    continue
        
        # Dynamic position sizing based on ATR volatility
        current_atr_pct = atr / price if price > 0 else 0
        if current_atr_pct > 0:
            size_multiplier = min(1.5, max(0.5, TARGET_ATR_PCT / current_atr_pct))
        else:
            size_multiplier = 1.0
        
        position_size = SIZE_FULL * size_multiplier
        position_size = min(SIZE_FULL, max(SIZE_HALF, position_size))  # Clamp
        
        # Entry logic
        if position_side == 0:
            if trend == 1:  # 4h uptrend
                # RSI pullback entry in uptrend
                if rsi_val < RSI_LONG_ENTRY and rsi_val > 30:
                    signals[i] = position_size
                    position_side = 1
                    entry_price = price
                    tp_triggered = False
                    highest_since_entry = price
                    lowest_since_entry = price
                    trailing_stop = price - ATR_STOP_MULT * atr
                    
            elif trend == -1:  # 4h downtrend
                # RSI rally entry in downtrend
                if rsi_val > RSI_SHORT_ENTRY and rsi_val < 70:
                    signals[i] = -position_size
                    position_side = -1
                    entry_price = price
                    tp_triggered = False
                    highest_since_entry = price
                    lowest_since_entry = price
                    trailing_stop = price + ATR_STOP_MULT * atr
            else:  # No clear trend
                signals[i] = 0.0
        else:
            # Hold existing position
            signals[i] = signals[i-1] if i > 0 else 0.0
            if position_side == 1 and signals[i] <= 0:
                signals[i] = SIZE_HALF if tp_triggered else SIZE_FULL
            elif position_side == -1 and signals[i] >= 0:
                signals[i] = -SIZE_HALF if tp_triggered else -SIZE_FULL
    
    return signals