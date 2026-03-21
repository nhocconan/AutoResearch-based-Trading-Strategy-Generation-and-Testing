#!/usr/bin/env python3
"""
EXPERIMENT #017 - Donchian Trend + RSI Pullback + ADX Strength Filter
====================================================================================
Hypothesis: Donchian breakouts capture trend direction more cleanly than MA crosses,
while RSI pullbacks provide better entry timing than MACD crosses. ADX filter ensures
we only trade when trend strength is sufficient (>25), avoiding choppy whipsaws.

Key differences from #016 (KAMA+MACD+BBW):
- Donchian(20) breakout trend instead of KAMA cross - cleaner trend signals
- RSI(14) pullback entries instead of MACD histogram - proven in #005 (Sharpe=5.525)
- ADX(14) strength filter instead of BBW regime - ensures strong trends only
- 4h Donchian trend + 1h RSI entries (MTF combo from #007 which got Sharpe=4.711)

Why this might beat Sharpe=5.525:
- Donchian breakouts catch trends earlier than MA crosses
- RSI pullbacks have proven entry timing (best strategy uses RSI)
- ADX > 25 filters out weak/choppy trends that cause losses
- Combines best elements from #005 (RSI) + #007 (Donchian)
"""

import numpy as np
import pandas as pd

name = "mtf_donchian_rsi_adx_v1"
timeframe = "1h"
leverage = 1.0


def calculate_donchian_channels(high, low, period=20):
    """
    Donchian Channels - highest high and lowest low over lookback period
    Returns upper channel, lower channel, and middle line
    """
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    middle = np.zeros(n)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
        middle[i] = (upper[i] + lower[i]) / 2
    
    return upper, lower, middle


def calculate_rsi(close, period=14):
    """
    Relative Strength Index - momentum oscillator (0-100)
    """
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
            rsi[i] = 100
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index - trend strength indicator (0-100)
    ADX > 25 indicates strong trend, < 20 indicates ranging market
    """
    n = len(close)
    adx = np.zeros(n)
    
    if n < period * 3:
        return adx
    
    # Calculate True Range and Directional Movement
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
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
    
    # Smooth TR, +DM, -DM using Wilder's method
    atr = np.zeros(n)
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    atr[period - 1] = np.mean(tr[1:period])
    plus_di[period - 1] = 100 * np.mean(plus_dm[1:period]) / atr[period - 1] if atr[period - 1] > 0 else 0
    minus_di[period - 1] = 100 * np.mean(minus_dm[1:period]) / atr[period - 1] if atr[period - 1] > 0 else 0
    
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        plus_di[i] = 100 * ((plus_dm[i] + plus_di[i - 1] * (period - 1)) / period) / atr[i] if atr[i] > 0 else 0
        minus_di[i] = 100 * ((minus_dm[i] + minus_di[i - 1] * (period - 1)) / period) / atr[i] if atr[i] > 0 else 0
    
    # Calculate DX and ADX
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        di_diff = abs(plus_di[i] - minus_di[i])
        if di_sum > 0:
            dx[i] = 100 * di_diff / di_sum
        else:
            dx[i] = 0
    
    # ADX is smoothed DX
    adx[period * 2 - 1] = np.mean(dx[period:period * 2])
    for i in range(period * 2, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    
    return adx


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
    adx_1h = calculate_adx(high, low, close, period=14)
    
    # 4h Donchian for trend filter (resample 1h → 4h)
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
    
    # Calculate 4h Donchian channels for trend
    donchian_upper_4h, donchian_lower_4h, donchian_mid_4h = calculate_donchian_channels(h_4h, l_4h, period=20)
    
    # 4h trend direction based on price position in Donchian channel
    trend_4h = np.zeros(len(c_4h))
    for i in range(20, len(c_4h)):
        if c_4h[i] > donchian_mid_4h[i] and c_4h[i] > donchian_upper_4h[i - 1]:
            trend_4h[i] = 1  # Bullish breakout
        elif c_4h[i] < donchian_mid_4h[i] and c_4h[i] < donchian_lower_4h[i - 1]:
            trend_4h[i] = -1  # Bearish breakout
        elif c_4h[i] > donchian_mid_4h[i]:
            trend_4h[i] = 1  # Above middle = bullish bias
        elif c_4h[i] < donchian_mid_4h[i]:
            trend_4h[i] = -1  # Below middle = bearish bias
    
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
    SIZE_FULL = 0.30   # Full position (conservative, max 0.40 per rules)
    SIZE_HALF = 0.15   # Half position (after take profit)
    
    # RSI thresholds for pullback entries
    RSI_LONG_ENTRY = 45  # Buy pullback in uptrend when RSI dips to 45
    RSI_SHORT_ENTRY = 55  # Sell pullback in downtrend when RSI rises to 55
    RSI_EXIT_LONG = 65  # Exit long when RSI gets overbought
    RSI_EXIT_SHORT = 35  # Exit short when RSI gets oversold
    
    # ADX filter - only trade when trend is strong
    ADX_MIN = 25  # Minimum ADX for strong trend
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    
    # Take profit multiplier (2R)
    TP_MULT = 2.0
    
    first_valid = max(50, 20, 42)  # Wait for all indicators (ADX needs 3*period)
    
    # Track entry prices for stoploss/takeprofit logic
    entry_price = np.zeros(n)
    position_side = np.zeros(n)  # 1 for long, -1 for short, 0 for flat
    tp_triggered = np.zeros(n)  # Track if take profit was hit
    highest_since_entry = np.zeros(n)  # For long positions
    lowest_since_entry = np.zeros(n)  # For short positions
    
    for i in range(first_valid, n):
        if np.isnan(rsi_1h[i]) or np.isnan(atr_1h[i]) or np.isnan(adx_1h[i]):
            signals[i] = 0.0
            continue
        
        trend = trend_1h[i]
        rsi = rsi_1h[i]
        adx = adx_1h[i]
        atr = atr_1h[i]
        price = close[i]
        
        # ATR filter - avoid trading when ATR is extremely high
        if atr > 0 and atr / price > 0.05:  # ATR > 5% of price = too volatile
            signals[i] = 0.0
            position_side[i] = 0
            entry_price[i] = 0
            tp_triggered[i] = 0
            continue
        
        # ADX filter - only trade when trend strength is sufficient
        if adx < ADX_MIN:
            # If in position, hold; otherwise stay flat
            if i > 0 and position_side[i - 1] != 0:
                signals[i] = signals[i - 1]
                position_side[i] = position_side[i - 1]
                entry_price[i] = entry_price[i - 1]
                tp_triggered[i] = tp_triggered[i - 1]
                highest_since_entry[i] = highest_since_entry[i - 1]
                lowest_since_entry[i] = lowest_since_entry[i - 1]
            else:
                signals[i] = 0.0
                position_side[i] = 0
            continue
        
        # Check trailing stop and take profit for existing positions
        if i > 0 and position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else price
            prev_tp = tp_triggered[i - 1]
            prev_highest = highest_since_entry[i - 1] if prev_side == 1 else 0
            prev_lowest = lowest_since_entry[i - 1] if prev_side == -1 else 0
            
            # Update highest/lowest since entry
            if prev_side == 1:
                current_highest = max(prev_highest, price)
                highest_since_entry[i] = current_highest
            elif prev_side == -1:
                current_lowest = min(prev_lowest, price)
                lowest_since_entry[i] = current_lowest
            
            if prev_side == 1:  # Long position
                # Trailing stoploss (2*ATR from entry or highest)
                stoploss_price = max(prev_entry, prev_highest) - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R)
                tp_price = prev_entry + TP_MULT * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    # Reduce to half position
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # RSI overbought exit
                if rsi > RSI_EXIT_LONG:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    continue
                
            elif prev_side == -1:  # Short position
                # Trailing stoploss (2*ATR from entry or lowest)
                stoploss_price = min(prev_entry, prev_lowest) + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R)
                tp_price = prev_entry - TP_MULT * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    # Reduce to half position
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # RSI oversold exit
                if rsi < RSI_EXIT_SHORT:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    lowest_since_entry[i] = 0
                    continue
            
            # Hold position if no exit signal
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            continue
        
        # RSI pullback entry signals (buy dips in uptrend, sell rallies in downtrend)
        if trend == 1:  # 4h uptrend
            # Long entry: RSI pulls back to 45 in uptrend
            if rsi <= RSI_LONG_ENTRY and rsi_1h[i - 1] > RSI_LONG_ENTRY:
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = 0
            else:
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
                tp_triggered[i] = 0
                highest_since_entry[i] = 0
                lowest_since_entry[i] = 0
                    
        elif trend == -1:  # 4h downtrend
            # Short entry: RSI rallies to 55 in downtrend
            if rsi >= RSI_SHORT_ENTRY and rsi_1h[i - 1] < RSI_SHORT_ENTRY:
                signals[i] = -SIZE_FULL
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = 0
                lowest_since_entry[i] = price
            else:
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
                tp_triggered[i] = 0
                highest_since_entry[i] = 0
                lowest_since_entry[i] = 0
        else:  # No clear trend
            signals[i] = 0.0
            position_side[i] = 0
            entry_price[i] = 0
            tp_triggered[i] = 0
            highest_since_entry[i] = 0
            lowest_since_entry[i] = 0
    
    return signals