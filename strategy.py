#!/usr/bin/env python3
"""
EXPERIMENT #026 - MTF HMA+MACD+RSI+BBW Regime Filter (1h+4h Optimized v1)
==================================================================================================
Hypothesis: Move from 15m to 1h entries for cleaner signals, add MACD momentum confirmation,
and use Bollinger Band Width for regime detection (avoid trading in low volatility chop).

Key changes from #025:
- Timeframe: 1h entries + 4h trend (vs 15m+4h) - reduces noise, better signal quality
- Add MACD histogram for momentum confirmation (entry only when MACD aligns with trend)
- Add BBW filter - only trade when BBW > 20th percentile (avoid low vol chop)
- Position size: 0.30 max (vs 0.35) - more conservative
- Stoploss: 2.0*ATR (vs 1.5*ATR) - gives trades more room, reduces premature exits
- RSI range: 40-60 (tighter than 35-65) - only take quality pullbacks

Why this should beat Sharpe=3.653:
- 1h timeframe has proven stable in historical tests
- MACD filter reduces false entries in weak trends
- BBW regime filter avoids choppy periods (major drawdown source)
- Conservative sizing (0.30) protects against deep DD
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_hma_macd_rsi_bbw_1h_4h_v1"
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
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    return atr


def calculate_hma(close, period=21):
    """Calculate Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).rolling(window=half, min_periods=half).apply(
        lambda x: np.sum(x * np.arange(1, half + 1)) / np.sum(np.arange(1, half + 1)), raw=True
    ).values
    
    wma2 = pd.Series(close).rolling(window=period, min_periods=period).apply(
        lambda x: np.sum(x * np.arange(1, period + 1)) / np.sum(np.arange(1, period + 1)), raw=True
    ).values
    
    raw_hma = 2 * wma1 - wma2
    
    hma = pd.Series(raw_hma).rolling(window=sqrt_period, min_periods=sqrt_period).apply(
        lambda x: np.sum(x * np.arange(1, sqrt_period + 1)) / np.sum(np.arange(1, sqrt_period + 1)), raw=True
    ).values
    
    return np.nan_to_num(hma)


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean().values
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean().values
    
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    rsi = 100 - (100 / (1 + rs))
    
    return np.nan_to_num(rsi)


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    n = len(close)
    if n < slow + signal:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    ema_fast = pd.Series(close).ewm(span=fast, adjust=False).mean().values
    ema_slow = pd.Series(close).ewm(span=slow, adjust=False).mean().values
    
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=signal, adjust=False).mean().values
    histogram = macd_line - signal_line
    
    return np.nan_to_num(macd_line), np.nan_to_num(signal_line), np.nan_to_num(histogram)


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Bandwidth"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = (upper - lower) / sma
    
    return np.nan_to_num(upper), np.nan_to_num(lower), np.nan_to_num(bandwidth)


def calculate_percentile_rank(series, window=100):
    """Calculate rolling percentile rank"""
    n = len(series)
    result = np.zeros(n)
    
    for i in range(window, n):
        window_data = series[i-window+1:i+1]
        count_below = np.sum(window_data <= series[i])
        result[i] = count_below / window
    
    return result


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Get 4h data for trend filter using mtf_data helper
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 1h indicators (entry timing)
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    hma_1h = calculate_hma(close, period=21)
    macd_1h, macd_sig_1h, macd_hist_1h = calculate_macd(close, fast=12, slow=26, signal=9)
    bb_upper_1h, bb_lower_1h, bbw_1h = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    # BBW percentile for regime filter
    bbw_pct_1h = calculate_percentile_rank(bbw_1h, window=100)
    
    # 4h indicators (trend filter) - using mtf_data helper
    hma_4h = calculate_hma(close_4h, period=21)
    macd_4h, macd_sig_4h, macd_hist_4h = calculate_macd(close_4h, fast=12, slow=26, signal=9)
    
    # Align 4h indicators to 1h timeframe
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    macd_hist_4h_aligned = align_htf_to_ltf(prices, df_4h, macd_hist_4h)
    close_4h_aligned = align_htf_to_ltf(prices, df_4h, close_4h)
    
    # Generate signals
    signals = np.zeros(n)
    
    # Position sizing parameters
    BASE_SIZE = 0.30
    TARGET_ATR_PCT = 0.025  # Target ATR as % of price
    
    # Entry thresholds
    RSI_LONG_MIN = 40
    RSI_LONG_MAX = 60
    RSI_SHORT_MIN = 40
    RSI_SHORT_MAX = 60
    
    # Stoploss
    ATR_STOP_MULT = 2.0
    
    # BBW regime filter - only trade when volatility is above 20th percentile
    BBW_MIN_PERCENTILE = 0.20
    
    first_valid = max(150, 40 * 4)  # Need enough 4h bars and BBW percentile
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        # Check for valid data
        if np.isnan(atr_1h[i]) or np.isnan(rsi_1h[i]) or atr_1h[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(macd_hist_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Get 4h trend
        trend_4h = 0
        if close_4h_aligned[i] > hma_4h_aligned[i]:
            trend_4h = 1
        elif close_4h_aligned[i] < hma_4h_aligned[i]:
            trend_4h = -1
        
        # 4h MACD momentum
        macd_momentum_4h = 0
        if macd_hist_4h_aligned[i] > 0:
            macd_momentum_4h = 1
        elif macd_hist_4h_aligned[i] < 0:
            macd_momentum_4h = -1
        
        # 1h MACD momentum
        macd_momentum_1h = 0
        if macd_hist_1h[i] > 0:
            macd_momentum_1h = 1
        elif macd_hist_1h[i] < 0:
            macd_momentum_1h = -1
        
        # ATR-based dynamic position sizing
        atr_pct = atr_1h[i] / close[i] if close[i] > 0 else 0
        if atr_pct > 0:
            size_multiplier = min(1.0, TARGET_ATR_PCT / atr_pct)
        else:
            size_multiplier = 1.0
        
        current_size = BASE_SIZE * size_multiplier
        current_size = min(current_size, 0.35)  # Cap at 0.35
        
        # Check stoploss and take profit for existing positions
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_tp = tp_triggered[i - 1]
            prev_high = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
            prev_low = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
            
            # Update highest/lowest since entry
            if prev_side == 1:
                current_high = max(prev_high, close[i])
                current_low = min(prev_low, close[i]) if prev_low > 0 else close[i]
            else:
                current_high = max(prev_high, close[i]) if prev_high > 0 else close[i]
                current_low = min(prev_low, close[i])
            
            highest_since_entry[i] = current_high
            lowest_since_entry[i] = current_low
            
            # Stoploss check (2.0*ATR)
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr_1h[i]
                if close[i] < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr_1h[i]
                if not prev_tp and close[i] >= tp_price:
                    signals[i] = current_size / 2
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr_1h[i]
                    if close[i] < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr_1h[i]
                if close[i] > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr_1h[i]
                if not prev_tp and close[i] <= tp_price:
                    signals[i] = -current_size / 2
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr_1h[i]
                    if close[i] > trail_stop:
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
        
        # Regime filter - only trade when BBW is above 20th percentile
        if bbw_pct_1h[i] < BBW_MIN_PERCENTILE:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Entry logic: 4h trend + 4h MACD + 1h RSI pullback + 1h MACD confirmation
        if trend_4h == 1 and macd_momentum_4h == 1:  # Bullish trend on 4h
            if (RSI_LONG_MIN <= rsi_1h[i] <= RSI_LONG_MAX and  # Pullback entry
                macd_momentum_1h >= 0):  # 1h MACD not bearish
                signals[i] = current_size
                position_side[i] = 1
                entry_price[i] = close[i]
                tp_triggered[i] = 0
                highest_since_entry[i] = close[i]
                lowest_since_entry[i] = close[i]
                
        elif trend_4h == -1 and macd_momentum_4h == -1:  # Bearish trend on 4h
            if (RSI_SHORT_MIN <= rsi_1h[i] <= RSI_SHORT_MAX and  # Pullback entry
                macd_momentum_1h <= 0):  # 1h MACD not bullish
                signals[i] = -current_size
                position_side[i] = -1
                entry_price[i] = close[i]
                tp_triggered[i] = 0
                highest_since_entry[i] = close[i]
                lowest_since_entry[i] = close[i]
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals