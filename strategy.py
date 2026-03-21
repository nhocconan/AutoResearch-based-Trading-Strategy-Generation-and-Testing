#!/usr/bin/env python3
"""
EXPERIMENT #009 - 4H Supertrend Trend + 1H RSI Pullback + Bollinger Band Width Regime
====================================================================================
Hypothesis: Combine 4h Supertrend for trend direction (volatility-adaptive trend follower)
with 1h RSI pullback entries (buy dips in uptrend, sell rallies in downtrend). Add
Bollinger Band Width regime filter to avoid trading during extreme volatility squeezes
or expansions. This combines proven elements from #005 (RSI pullback) and #007 (Donchian)
but uses Supertrend instead for better volatility adaptation.

Why this might beat Sharpe=5.525:
- Supertrend adapts stop levels based on ATR (better than fixed EMA distance)
- RSI pullback entries capture mean-reversion within trends (proven in #005/#007)
- Bollinger Band Width filter identifies regime (avoid squeeze breakouts that fail)
- Multi-timeframe: 4h trend + 1h entries (proven architecture)
- Discrete signal levels (0.0, ±0.20, ±0.35) minimize churn costs
- ATR-based stoploss protects against adverse moves
"""

import numpy as np
import pandas as pd

name = "mtf_supertrend_rsi_bbw_v1"
timeframe = "1h"
leverage = 1.0


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Calculate Supertrend indicator
    Supertrend = (HL2) ± (multiplier * ATR)
    where HL2 = (high + low) / 2
    Trend flips when price crosses the supertrend line
    """
    n = len(close)
    supertrend = np.zeros(n)
    direction = np.zeros(n)  # 1 for bullish, -1 for bearish
    
    # Calculate ATR
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
    
    # Calculate Supertrend
    hl2 = (high + low) / 2.0
    
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    
    for i in range(period, n):
        upper_band[i] = hl2[i] + multiplier * atr[i]
        lower_band[i] = hl2[i] - multiplier * atr[i]
    
    # Initialize
    supertrend[period] = upper_band[period]
    direction[period] = -1  # Start bearish
    
    for i in range(period + 1, n):
        if direction[i - 1] == 1:  # Previous was bullish
            if close[i] > lower_band[i]:
                supertrend[i] = lower_band[i]
                direction[i] = 1
            else:
                supertrend[i] = upper_band[i]
                direction[i] = -1
        else:  # Previous was bearish
            if close[i] < upper_band[i]:
                supertrend[i] = upper_band[i]
                direction[i] = -1
            else:
                supertrend[i] = lower_band[i]
                direction[i] = 1
    
    return supertrend, direction, atr


def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing method"""
    n = len(close)
    rsi = np.zeros(n)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    for i in range(1, n):
        if delta[i - 1] > 0:
            gain[i] = delta[i - 1]
        else:
            loss[i] = -delta[i - 1]
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    
    avg_gain[period] = np.mean(gain[1:period + 1])
    avg_loss[period] = np.mean(loss[1:period + 1])
    
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
    
    for i in range(period, n):
        if avg_loss[i] == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Band Width"""
    n = len(close)
    upper = np.zeros(n)
    middle = np.zeros(n)
    lower = np.zeros(n)
    bandwidth = np.zeros(n)
    percent_b = np.zeros(n)
    
    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
        middle[i] = np.mean(window)
        std = np.std(window)
        upper[i] = middle[i] + std_mult * std
        lower[i] = middle[i] - std_mult * std
        
        if middle[i] > 0:
            bandwidth[i] = (upper[i] - lower[i]) / middle[i]
        else:
            bandwidth[i] = 0
        
        if upper[i] != lower[i]:
            percent_b[i] = (close[i] - lower[i]) / (upper[i] - lower[i])
        else:
            percent_b[i] = 0.5
    
    return upper, middle, lower, bandwidth, percent_b


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
    atr[period - 1] = np.mean(tr[1:period])
    
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    return atr


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values if "volume" in prices.columns else np.ones(len(close))
    n = len(close)
    
    # 1h indicators for entry timing
    rsi_1h = calculate_rsi(close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    bb_upper_1h, bb_mid_1h, bb_lower_1h, bb_bw_1h, bb_pct_1h = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    # 4h indicators for trend (resample 1h → 4h)
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
    
    c_4h = df_4h['close'].values
    h_4h = df_4h['high'].values
    l_4h = df_4h['low'].values
    n_4h = len(c_4h)
    
    # Calculate 4h Supertrend
    supertrend_4h, direction_4h, atr_4h = calculate_supertrend(h_4h, l_4h, c_4h, period=10, multiplier=3.0)
    
    # Calculate 4h Bollinger Band Width for regime filter
    _, _, _, bb_bw_4h, _ = calculate_bollinger_bands(c_4h, period=20, std_mult=2.0)
    
    # Map 4h indicators back to 1h timeframe (4 x 1h = 4h)
    trend_1h = np.zeros(n)
    bb_bw_1h_from_4h = np.zeros(n)
    idx_1h_to_4h = np.arange(n) // 4
    
    for i in range(n):
        idx_4h = idx_1h_to_4h[i]
        if idx_4h < len(direction_4h):
            trend_1h[i] = direction_4h[idx_4h]
            bb_bw_1h_from_4h[i] = bb_bw_4h[idx_4h]
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels to reduce churn
    SIZE_FULL = 0.35   # Full position
    SIZE_HALF = 0.20   # Half position (after take profit)
    
    # RSI thresholds for pullback entries
    RSI_LONG_ENTRY = 45.0    # Buy when RSI pulls back to 45 in uptrend
    RSI_SHORT_ENTRY = 55.0   # Sell when RSI rallies to 55 in downtrend
    RSI_EXIT_LONG = 65.0     # Exit long when RSI reaches 65 (overbought)
    RSI_EXIT_SHORT = 35.0    # Exit short when RSI reaches 35 (oversold)
    
    # Bollinger Band Width regime filter
    BBW_MIN = 0.02    # Minimum bandwidth to trade (avoid extreme squeeze)
    BBW_MAX = 0.15    # Maximum bandwidth to trade (avoid extreme expansion)
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    
    # Take profit multiplier (2R)
    TP_MULT = 2.0
    
    first_valid = max(60, 30, 14, 28, 50)
    
    # Track entry prices for stoploss/takeprofit logic
    entry_price = np.zeros(n)
    position_side = np.zeros(n)  # 1 for long, -1 for short, 0 for flat
    tp_triggered = np.zeros(n)  # Track if take profit was hit
    
    # Track highest/lowest price since entry for trailing stop
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(rsi_1h[i]) or np.isnan(atr_1h[i]) or np.isnan(bb_bw_1h_from_4h[i]):
            signals[i] = 0.0
            continue
        
        trend = trend_1h[i]
        rsi_val = rsi_1h[i]
        atr = atr_1h[i]
        price = close[i]
        bbw_4h = bb_bw_1h_from_4h[i]
        
        # Bollinger Band Width regime filter - avoid extreme volatility regimes
        if bbw_4h < BBW_MIN or bbw_4h > BBW_MAX:
            if i > 0 and position_side[i - 1] != 0:
                signals[i] = signals[i - 1]
                position_side[i] = position_side[i - 1]
                entry_price[i] = entry_price[i - 1]
                tp_triggered[i] = tp_triggered[i - 1]
                highest_since_entry[i] = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else price
                lowest_since_entry[i] = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else price
            else:
                signals[i] = 0.0
                position_side[i] = 0
            continue
        
        # ATR filter - avoid trading when ATR is extremely high (>5% of price)
        if atr > 0 and atr / price > 0.05:
            signals[i] = 0.0
            position_side[i] = 0
            entry_price[i] = 0
            tp_triggered[i] = 0
            continue
        
        # Update highest/lowest since entry for existing positions
        if i > 0 and position_side[i - 1] != 0:
            highest_since_entry[i] = max(highest_since_entry[i - 1], price)
            lowest_since_entry[i] = min(lowest_since_entry[i - 1], price)
        else:
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        
        # Check trailing stop and take profit for existing positions
        if i > 0 and position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else price
            prev_tp = tp_triggered[i - 1]
            
            if prev_side == 1:
                # Stoploss check (2*ATR against position)
                stoploss_price = prev_entry - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    continue
                
                # Trailing stop: move stop up as price makes new highs
                trailing_stop = highest_since_entry[i] - ATR_STOP_MULT * atr
                
                # Take profit check (2R)
                tp_price = prev_entry + TP_MULT * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # RSI overbought exit
                if rsi_val >= RSI_EXIT_LONG:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    continue
                
                # Trend reversal exit (Supertrend flips bearish)
                if trend == -1:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    continue
                
            elif prev_side == -1:
                # Stoploss check (2*ATR against position)
                stoploss_price = prev_entry + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    continue
                
                # Trailing stop: move stop down as price makes new lows
                trailing_stop = lowest_since_entry[i] + ATR_STOP_MULT * atr
                
                # Take profit check (2R)
                tp_price = prev_entry - TP_MULT * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # RSI oversold exit
                if rsi_val <= RSI_EXIT_SHORT:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    continue
                
                # Trend reversal exit (Supertrend flips bullish)
                if trend == 1:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    continue
        
        # Generate new entries based on trend + RSI pullback
        if trend == 1:  # 4h uptrend (Supertrend bullish)
            # RSI pullback to 45 (buy the dip in uptrend)
            if rsi_val <= RSI_LONG_ENTRY:
                if position_side[i - 1] != -1:
                    signals[i] = SIZE_FULL
                    position_side[i] = 1
                    entry_price[i] = price
                    tp_triggered[i] = 0
                    highest_since_entry[i] = price
                    lowest_since_entry[i] = price
            else:
                if i > 0 and position_side[i - 1] == 1:
                    signals[i] = signals[i - 1]
                    position_side[i] = 1
                    entry_price[i] = entry_price[i - 1]
                    tp_triggered[i] = tp_triggered[i - 1]
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    
        elif trend == -1:  # 4h downtrend (Supertrend bearish)
            # RSI rally to 55 (sell the rip in downtrend)
            if rsi_val >= RSI_SHORT_ENTRY:
                if position_side[i - 1] != 1:
                    signals[i] = -SIZE_FULL
                    position_side[i] = -1
                    entry_price[i] = price
                    tp_triggered[i] = 0
                    highest_since_entry[i] = price
                    lowest_since_entry[i] = price
            else:
                if i > 0 and position_side[i - 1] == -1:
                    signals[i] = signals[i - 1]
                    position_side[i] = -1
                    entry_price[i] = entry_price[i - 1]
                    tp_triggered[i] = tp_triggered[i - 1]
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
        else:
            signals[i] = 0.0
            position_side[i] = 0
            entry_price[i] = 0
            tp_triggered[i] = 0
    
    return signals