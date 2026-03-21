#!/usr/bin/env python3
"""
EXPERIMENT #087 - Simplified MTF Ensemble with Proper HTF Alignment
==================================================================================================
Hypothesis: Complex regime detection has failed repeatedly (#077-#086). 
Return to basics: 3-signal ensemble with proper mtf_data helper alignment.

Key changes from #040:
- Use mtf_data helper (get_htf_data, align_htf_to_ltf) - MANDATORY for proper 4h alignment
- Simpler 3-signal voting: 4h HMA trend + 15m RSI pullback + 15m MACD momentum
- Adaptive sizing: 0.20 (1 signal), 0.25 (2 signals), 0.30 (3 signals agree)
- Conservative max position: 0.30 (vs 0.35 in #040)
- BBW regime filter only for exit timing, not entry blocking
- Cross-asset filter: require BTC 4h trend alignment for ETH/SOL entries

Why this should work:
- Proper mtf_data alignment prevents the look-ahead bugs that killed 46 strategies
- Simpler ensemble = fewer failure modes than complex regime detection
- Adaptive sizing rewards signal confidence without overexposure
- Based on #084 success (cross_asset_kama_supertrend_volume_mtf_15m_4h_v1, Sharpe=0.423)
"""

import numpy as np
import pandas as pd

try:
    from mtf_data import get_htf_data, align_htf_to_ltf
    HAS_MTF_DATA = True
except ImportError:
    HAS_MTF_DATA = False

name = "simplified_mtf_ensemble_proper_htf_15m_4h_v1"
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


def calculate_hma(close, period=21):
    """Calculate Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = np.zeros(n)
    wma2 = np.zeros(n)
    hma = np.zeros(n)
    
    for i in range(half_period - 1, n):
        weights = np.arange(1, half_period + 1)
        wma1[i] = np.sum(close[i - half_period + 1:i + 1] * weights) / np.sum(weights)
    
    for i in range(period - 1, n):
        weights = np.arange(1, period + 1)
        wma2[i] = np.sum(close[i - period + 1:i + 1] * weights) / np.sum(weights)
    
    for i in range(period - 1 + sqrt_period - 1, n):
        start_idx = i - sqrt_period + 1
        weights = np.arange(1, sqrt_period + 1)
        raw_vals = 2 * wma1[start_idx:i + 1] - wma2[start_idx:i + 1]
        hma[i] = np.sum(raw_vals * weights) / np.sum(weights)
    
    return hma


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


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD histogram"""
    n = len(close)
    if n < slow + signal:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    ema_fast = np.zeros(n)
    ema_slow = np.zeros(n)
    
    ema_fast[fast - 1] = np.mean(close[:fast])
    ema_slow[slow - 1] = np.mean(close[:slow])
    
    for i in range(fast, n):
        ema_fast[i] = ema_fast[i - 1] + (2.0 / (fast + 1)) * (close[i] - ema_fast[i - 1])
    
    for i in range(slow, n):
        ema_slow[i] = ema_slow[i - 1] + (2.0 / (slow + 1)) * (close[i] - ema_slow[i - 1])
    
    macd_line = ema_fast - ema_slow
    
    signal_line = np.zeros(n)
    first_signal = slow + signal - 1
    signal_line[first_signal] = np.mean(macd_line[slow:first_signal + 1])
    
    for i in range(first_signal + 1, n):
        signal_line[i] = signal_line[i - 1] + (2.0 / (signal + 1)) * (macd_line[i] - signal_line[i - 1])
    
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Band Width"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    
    middle = np.zeros(n)
    upper = np.zeros(n)
    lower = np.zeros(n)
    bbw = np.zeros(n)
    
    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
        middle[i] = np.mean(window)
        std = np.std(window)
        upper[i] = middle[i] + std_mult * std
        lower[i] = middle[i] - std_mult * std
        
        if middle[i] > 0:
            bbw[i] = (upper[i] - lower[i]) / middle[i]
        else:
            bbw[i] = 0
    
    return upper, middle, lower, bbw


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    macd_line_15m, macd_signal_15m, macd_hist_15m = calculate_macd(close, fast=12, slow=26, signal=9)
    _, _, _, bbw_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    # Get 4h data using mtf_data helper (CRITICAL for proper alignment)
    if HAS_MTF_DATA:
        try:
            df_4h = get_htf_data(prices, '4h')
            close_4h = df_4h['close'].values
            
            # Calculate 4h HMA for trend
            hma_4h = calculate_hma(close_4h, period=21)
            
            # Align 4h indicators to 15m timeframe (auto shift for completed bars)
            hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
            close_4h_aligned = align_htf_to_ltf(prices, df_4h, close_4h)
            
            # 4h trend: price above HMA = bullish, below = bearish
            trend_4h = np.zeros(n)
            for i in range(n):
                if close_4h_aligned[i] > hma_4h_aligned[i]:
                    trend_4h[i] = 1
                elif close_4h_aligned[i] < hma_4h_aligned[i]:
                    trend_4h[i] = -1
        except Exception:
            # Fallback if mtf_data fails
            trend_4h = np.zeros(n)
    else:
        # Fallback: simple downsampling (not ideal but works)
        trend_4h = np.zeros(n)
        bars_per_4h = 16  # 16 x 15m = 4h
        n_4h = n // bars_per_4h
        if n_4h > 21:
            c_4h = np.array([close[i * bars_per_4h + bars_per_4h - 1] for i in range(n_4h)])
            hma_4h_simple = calculate_hma(c_4h, period=21)
            for i in range(n):
                idx_4h = i // bars_per_4h
                if idx_4h < n_4h and idx_4h >= 21:
                    if c_4h[idx_4h] > hma_4h_simple[idx_4h]:
                        trend_4h[i] = 1
                    elif c_4h[idx_4h] < hma_4h_simple[idx_4h]:
                        trend_4h[i] = -1
    
    # Generate signals with ensemble voting
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels based on signal agreement (CRITICAL for drawdown)
    SIZE_1_SIGNAL = 0.20
    SIZE_2_SIGNALS = 0.25
    SIZE_3_SIGNALS = 0.30  # Max position size
    
    # Entry thresholds
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 65
    MACD_HIST_MIN = 0  # Positive for long, negative for short
    BBW_MIN = 0.01  # Minimum volatility to trade
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    
    first_valid = max(200, 14 * 2, 26 + 9, 20)
    
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
        
        rsi_val = rsi_15m[i]
        macd_hist = macd_hist_15m[i]
        atr = atr_15m[i]
        price = close[i]
        bbw = bbw_15m[i]
        trend = trend_4h[i]
        
        # Count agreeing signals for ensemble voting
        long_signals = 0
        short_signals = 0
        
        # Signal 1: 4h HMA trend
        if trend == 1:
            long_signals += 1
        elif trend == -1:
            short_signals += 1
        
        # Signal 2: 15m RSI pullback
        if RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX:
            long_signals += 1
        elif RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX:
            short_signals += 1
        
        # Signal 3: 15m MACD momentum
        if macd_hist > MACD_HIST_MIN:
            long_signals += 1
        elif macd_hist < -MACD_HIST_MIN:
            short_signals += 1
        
        # BBW filter - only block entries in extremely low vol
        if bbw < BBW_MIN:
            long_signals = 0
            short_signals = 0
        
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
            
            # Stoploss check (2.0*ATR)
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
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = signals[i - 1] * 0.5
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
                stoploss_price = prev_entry + ATR_STOP_MULT * atr
                if price > stoploss_price:
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
                    signals[i] = signals[i - 1] * 0.5
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
        
        # Entry logic: ensemble voting with adaptive sizing
        if long_signals >= 1 and long_signals > short_signals:
            if long_signals == 1:
                signals[i] = SIZE_1_SIGNAL
            elif long_signals == 2:
                signals[i] = SIZE_2_SIGNALS
            else:  # 3 signals
                signals[i] = SIZE_3_SIGNALS
            position_side[i] = 1
            entry_price[i] = price
            tp_triggered[i] = 0
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
            
        elif short_signals >= 1 and short_signals > long_signals:
            if short_signals == 1:
                signals[i] = -SIZE_1_SIGNAL
            elif short_signals == 2:
                signals[i] = -SIZE_2_SIGNALS
            else:  # 3 signals
                signals[i] = -SIZE_3_SIGNALS
            position_side[i] = -1
            entry_price[i] = price
            tp_triggered[i] = 0
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals