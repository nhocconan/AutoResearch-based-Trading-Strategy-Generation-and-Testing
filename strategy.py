#!/usr/bin/env python3
"""
EXPERIMENT #012 - MTF Supertrend+MACD+RSI+BBW (15m+1h+4h v1)
==================================================================================================
Hypothesis: Combine proven Supertrend (4H trend) with MACD histogram (1H momentum) + RSI (15m entry).
BBW filter avoids choppy sideways markets. This differs from #009 by using Supertrend on 4H 
instead of KAMA, and adding MACD histogram for momentum confirmation.

Why this should work:
- Supertrend is excellent for trend direction (used in best strategy #009)
- MACD histogram provides early momentum shifts before price moves
- RSI pullback entries work well in trending markets
- BBW filter avoids low-volatility chop (proven in #009)
- Three timeframes maintain the multi-timeframe advantage

Key differences from #009:
- Supertrend on 4H instead of KAMA
- MACD histogram on 1H for momentum (new)
- Simpler position tracking for speed
- Same proven BBW + RSI combination
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_supertrend_macd_rsi_bbw_15m_1h_4h_v1"
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
    
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    supertrend = np.zeros(n)
    direction = np.ones(n)  # 1 = bullish, -1 = bearish
    
    for i in range(period, n):
        if atr[i] == 0:
            upper_band[i] = close[i]
            lower_band[i] = close[i]
        else:
            upper_band[i] = (high[i] + low[i]) / 2 + multiplier * atr[i]
            lower_band[i] = (high[i] + low[i]) / 2 - multiplier * atr[i]
        
        if i == period:
            supertrend[i] = upper_band[i]
            direction[i] = -1
        else:
            if close[i - 1] > supertrend[i - 1]:
                supertrend[i] = lower_band[i] if lower_band[i] < supertrend[i - 1] else supertrend[i - 1]
                direction[i] = 1
            else:
                supertrend[i] = upper_band[i] if upper_band[i] > supertrend[i - 1] else supertrend[i - 1]
                direction[i] = -1
    
    return supertrend, direction


def calculate_ema(close, period):
    """Calculate Exponential Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    ema = pd.Series(close).ewm(span=period, adjust=False).mean().values
    return ema


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD (histogram, macd_line, signal_line)"""
    n = len(close)
    if n < slow:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    ema_fast = pd.Series(close).ewm(span=fast, adjust=False).mean().values
    ema_slow = pd.Series(close).ewm(span=slow, adjust=False).mean().values
    
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=signal, adjust=False).mean().values
    histogram = macd_line - signal_line
    
    return histogram, macd_line, signal_line


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False).mean().values
    
    rs = np.zeros(n)
    for i in range(n):
        if avg_loss[i] > 0:
            rs[i] = avg_gain[i] / avg_loss[i]
        else:
            rs[i] = 100
    
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Band Width"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bb_width = (upper - lower) / sma * 100  # Band width as percentage
    
    return upper, lower, bb_width


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    bb_upper, bb_lower, bbw_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    # Get 1h data using mtf_data helper
    try:
        df_1h = get_htf_data(prices, '1h')
        c_1h = df_1h['close'].values
        
        # 1h MACD for momentum
        macd_hist_1h, macd_line_1h, macd_sig_1h = calculate_macd(c_1h, fast=12, slow=26, signal=9)
        
        # Align 1h indicators to 15m timeframe
        macd_hist_1h_aligned = align_htf_to_ltf(prices, df_1h, macd_hist_1h)
    except Exception:
        macd_hist_1h_aligned = np.zeros(n)
    
    # Get 4h data using mtf_data helper for trend filter
    try:
        df_4h = get_htf_data(prices, '4h')
        c_4h = df_4h['close'].values
        h_4h = df_4h['high'].values
        l_4h = df_4h['low'].values
        
        # 4h Supertrend for trend direction
        supertrend_4h, st_direction_4h = calculate_supertrend(h_4h, l_4h, c_4h, period=10, multiplier=3.0)
        
        # Align 4h indicators to 15m timeframe
        st_direction_4h_aligned = align_htf_to_ltf(prices, df_4h, st_direction_4h)
    except Exception:
        st_direction_4h_aligned = np.zeros(n)
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.30  # Conservative position size
    SIZE_HALF = 0.15
    
    # RSI thresholds for entries
    RSI_LONG_ENTRY = 45  # Pullback in uptrend
    RSI_SHORT_ENTRY = 55  # Pullback in downtrend
    
    # BBW threshold - avoid choppy markets
    BBW_MIN = 3.0  # Minimum band width percentage
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    
    # MACD histogram threshold
    MACD_MIN = 0.0  # Must be positive for long, negative for short
    
    first_valid = max(200, 26, 14 * 2, 20)
    
    # Track position state (simplified for speed)
    position_side = 0
    entry_price = 0.0
    tp_triggered = False
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    stoploss_price = 0.0
    trail_stop_price = 0.0
    
    for i in range(first_valid, n):
        # Skip invalid data
        if np.isnan(atr_15m[i]) or atr_15m[i] == 0 or np.isnan(rsi_15m[i]):
            signals[i] = 0.0
            continue
        
        # Get aligned MTF values
        macd_hist_1h = macd_hist_1h_aligned[i] if i < len(macd_hist_1h_aligned) else 0.0
        st_dir_4h = st_direction_4h_aligned[i] if i < len(st_direction_4h_aligned) else 0.0
        
        # BBW filter - avoid choppy markets
        if bbw_15m[i] < BBW_MIN:
            if position_side != 0:
                # Close existing position
                signals[i] = 0.0
                position_side = 0
                entry_price = 0.0
                tp_triggered = False
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # 4h Supertrend trend filter
        if st_dir_4h == 1:
            trend_4h = 1  # Bullish
        elif st_dir_4h == -1:
            trend_4h = -1  # Bearish
        else:
            trend_4h = 0  # Neutral
        
        if trend_4h == 0:
            if position_side != 0:
                signals[i] = 0.0
                position_side = 0
                entry_price = 0.0
                tp_triggered = False
            else:
                signals[i] = 0.0
            continue
        
        # Check existing positions first (stoploss/TP management)
        if position_side != 0:
            # Update highest/lowest since entry
            highest_since_entry = max(highest_since_entry, close[i])
            lowest_since_entry = min(lowest_since_entry, close[i])
            
            # Stoploss check (2.0*ATR)
            if position_side == 1:
                if close[i] < stoploss_price:
                    signals[i] = 0.0
                    position_side = 0
                    entry_price = 0.0
                    tp_triggered = False
                    highest_since_entry = 0.0
                    lowest_since_entry = 0.0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = entry_price + 2 * ATR_STOP_MULT * atr_15m[i]
                if not tp_triggered and close[i] >= tp_price:
                    signals[i] = SIZE_HALF
                    tp_triggered = True
                    # Set trail stop at 1R
                    trail_stop_price = entry_price + ATR_STOP_MULT * atr_15m[i]
                    continue
                
                # Trail stop at 1R profit after TP
                if tp_triggered:
                    trail_stop_price = max(trail_stop_price, highest_since_entry - ATR_STOP_MULT * atr_15m[i])
                    if close[i] < trail_stop_price:
                        signals[i] = 0.0
                        position_side = 0
                        entry_price = 0.0
                        tp_triggered = False
                        highest_since_entry = 0.0
                        lowest_since_entry = 0.0
                        continue
            
            elif position_side == -1:
                if close[i] > stoploss_price:
                    signals[i] = 0.0
                    position_side = 0
                    entry_price = 0.0
                    tp_triggered = False
                    highest_since_entry = 0.0
                    lowest_since_entry = 0.0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = entry_price - 2 * ATR_STOP_MULT * atr_15m[i]
                if not tp_triggered and close[i] <= tp_price:
                    signals[i] = -SIZE_HALF
                    tp_triggered = True
                    # Set trail stop at 1R
                    trail_stop_price = entry_price - ATR_STOP_MULT * atr_15m[i]
                    continue
                
                # Trail stop at 1R profit after TP
                if tp_triggered:
                    trail_stop_price = min(trail_stop_price, lowest_since_entry + ATR_STOP_MULT * atr_15m[i])
                    if close[i] > trail_stop_price:
                        signals[i] = 0.0
                        position_side = 0
                        entry_price = 0.0
                        tp_triggered = False
                        highest_since_entry = 0.0
                        lowest_since_entry = 0.0
                        continue
            
            # Hold position if no exit triggered
            signals[i] = SIZE_FULL if position_side == 1 else -SIZE_HALF if tp_triggered else -SIZE_FULL
            continue
        
        # Entry logic: 4h Supertrend + 1h MACD + 15m RSI + BBW
        if trend_4h == 1:  # Bullish trend on 4h
            # 1h MACD histogram positive (momentum)
            # 15m RSI pullback entry (not overbought)
            if (macd_hist_1h > MACD_MIN and
                rsi_15m[i] < RSI_LONG_ENTRY + 15 and rsi_15m[i] > RSI_LONG_ENTRY - 15):
                signals[i] = SIZE_FULL
                position_side = 1
                entry_price = close[i]
                tp_triggered = False
                highest_since_entry = close[i]
                lowest_since_entry = close[i]
                stoploss_price = entry_price - ATR_STOP_MULT * atr_15m[i]
            else:
                signals[i] = 0.0
                
        elif trend_4h == -1:  # Bearish trend on 4h
            # 1h MACD histogram negative (momentum)
            # 15m RSI pullback entry (not oversold)
            if (macd_hist_1h < -MACD_MIN and
                rsi_15m[i] > RSI_SHORT_ENTRY - 15 and rsi_15m[i] < RSI_SHORT_ENTRY + 15):
                signals[i] = -SIZE_FULL
                position_side = -1
                entry_price = close[i]
                tp_triggered = False
                highest_since_entry = close[i]
                lowest_since_entry = close[i]
                stoploss_price = entry_price + ATR_STOP_MULT * atr_15m[i]
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals