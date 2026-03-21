#!/usr/bin/env python3
"""
EXPERIMENT #062 - Adaptive KAMA-Supertrend Regime Strategy with Clean State Management
==================================================================================================
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts better to volatility regimes than HMA/EMA.
Combine 4h KAMA-Supertrend trend filter with 15m RSI-MACD entries, using BBW for regime detection.

Key differences from #061:
- KAMA instead of HMA (better adaptation to volatility changes)
- Cleaner entry logic: only enter when BOTH trend AND momentum agree
- Discrete signal levels: 0.0, ±0.20, ±0.30 (reduce churn costs)
- Regime-adaptive: trend-follow in low vol, reduced size in high vol
- Proper ATR-based stoploss (2.5*ATR) with trend-reversal exit
- No complex TP trailing (simpler = fewer bugs)

Why this should beat Sharpe=0.223:
- KAMA adapts to market efficiency (ER) - better in choppy markets
- Proven 4h+15m combination from baseline (Sharpe=3.653)
- Conservative sizing (max 0.30) controls drawdown
- Fewer signal changes = lower fee drag
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "adaptive_kama_supertrend_regime_15m_4h_v1"
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


def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA)
    KAMA adapts to market noise using Efficiency Ratio (ER)
    """
    n = len(close)
    if n < period + slow:
        return np.zeros(n)
    
    kama = np.zeros(n)
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        signal = abs(close[i] - close[i - period])
        noise = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    # Calculate smoothing constant
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    
    # Initialize KAMA
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
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


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    trend_direction = np.ones(n)
    
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    
    for i in range(period, n):
        mid = (high[i] + low[i]) / 2
        upper_band[i] = mid + multiplier * atr[i]
        lower_band[i] = mid - multiplier * atr[i]
    
    supertrend[period] = lower_band[period]
    
    for i in range(period + 1, n):
        if trend_direction[i - 1] == 1:
            supertrend[i] = max(lower_band[i], supertrend[i - 1])
            if close[i] < supertrend[i]:
                supertrend[i] = upper_band[i]
                trend_direction[i] = -1
            else:
                trend_direction[i] = 1
        else:
            supertrend[i] = min(upper_band[i], supertrend[i - 1])
            if close[i] > supertrend[i]:
                supertrend[i] = lower_band[i]
                trend_direction[i] = 1
            else:
                trend_direction[i] = -1
    
    return supertrend, trend_direction


def calculate_macd_histogram(close, fast=12, slow=26, signal=9):
    """Calculate MACD histogram only"""
    n = len(close)
    if n < slow + signal:
        return np.zeros(n)
    
    def ema(data, period):
        result = np.zeros(n)
        multiplier = 2 / (period + 1)
        result[period - 1] = np.mean(data[:period])
        for i in range(period, n):
            result[i] = (data[i] - result[i - 1]) * multiplier + result[i - 1]
        return result
    
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    
    signal_line = np.zeros(n)
    signal_line[slow + signal - 1] = np.mean(macd_line[slow:slow + signal])
    for i in range(slow + signal, n):
        signal_line[i] = (macd_line[i] - signal_line[i - 1]) * (2 / (signal + 1)) + signal_line[i - 1]
    
    histogram = macd_line - signal_line
    return histogram


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


def calculate_bbw_percentile(bbw, lookback=200):
    """Calculate BBW percentile for regime detection"""
    n = len(bbw)
    percentile = np.zeros(n)
    
    for i in range(lookback - 1, n):
        window = bbw[i - lookback + 1:i + 1]
        rank = np.sum(window <= bbw[i])
        percentile[i] = rank / lookback
    
    return percentile


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    macd_hist_15m = calculate_macd_histogram(close, fast=12, slow=26, signal=9)
    _, _, _, bbw_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    bbw_pct_15m = calculate_bbw_percentile(bbw_15m, lookback=200)
    
    # Get 4h data using mtf_data helper (CRITICAL - proper alignment)
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        # Calculate 4h indicators
        kama_4h = calculate_kama(close_4h, period=10, fast=2, slow=30)
        _, st_direction_4h = calculate_supertrend(high_4h, low_4h, close_4h, period=10, multiplier=3.0)
        
        # Align 4h indicators to 15m timeframe (auto shift for completed bars)
        kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h)
        st_direction_4h_aligned = align_htf_to_ltf(prices, df_4h, st_direction_4h)
        
        mtf_available = True
    except Exception:
        mtf_available = False
        kama_4h_aligned = np.zeros(n)
        st_direction_4h_aligned = np.ones(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_LOW_VOL = 0.30  # Low volatility regime (trend follow)
    SIZE_HIGH_VOL = 0.20  # High volatility regime (reduced risk)
    
    # Signal thresholds
    RSI_LONG_MIN = 45
    RSI_LONG_MAX = 65
    RSI_SHORT_MIN = 35
    RSI_SHORT_MAX = 55
    MACD_THRESHOLD = 0
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5
    
    # BBW percentile for regime detection
    BBW_HIGH_VOL_PCT = 0.70  # Above this = high volatility regime
    
    first_valid = max(250, 14 * 2, 20, 200)
    
    # Generate signals
    signals = np.zeros(n)
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    
    for i in range(first_valid, n):
        if np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or np.isnan(macd_hist_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            continue
        
        # Regime detection
        bbw_pct = bbw_pct_15m[i]
        high_vol_regime = bbw_pct > BBW_HIGH_VOL_PCT
        
        # Set position size based on regime
        if high_vol_regime:
            current_size = SIZE_HIGH_VOL
        else:
            current_size = SIZE_LOW_VOL
        
        # === 4h Trend Signal ===
        trend_signal = 0
        if mtf_available and kama_4h_aligned[i] > 0:
            price_4h = close_4h[min(i // 16, len(close_4h) - 1)] if len(close_4h) > 0 else close[i]
            kama_4h_val = kama_4h_aligned[i]
            st_trend_4h = st_direction_4h_aligned[i]
            
            # Bullish: price above KAMA + supertrend up
            if price_4h > kama_4h_val and st_trend_4h == 1:
                trend_signal = 1
            # Bearish: price below KAMA + supertrend down
            elif price_4h < kama_4h_val and st_trend_4h == -1:
                trend_signal = -1
        
        # === 15m Entry Signal ===
        entry_signal = 0
        rsi_15m_val = rsi_15m[i]
        macd_15m_val = macd_hist_15m[i]
        
        # Long entry: RSI in neutral-bullish zone + MACD positive
        if RSI_LONG_MIN <= rsi_15m_val <= RSI_LONG_MAX and macd_15m_val > MACD_THRESHOLD:
            entry_signal = 1
        # Short entry: RSI in neutral-bearish zone + MACD negative
        elif RSI_SHORT_MIN <= rsi_15m_val <= RSI_SHORT_MAX and macd_15m_val < MACD_THRESHOLD:
            entry_signal = -1
        
        # === Check existing position for stoploss/exit ===
        if in_position:
            # Calculate stoploss level
            if position_side == 1:
                stoploss_price = entry_price - ATR_STOP_MULT * entry_atr
                if close[i] < stoploss_price:
                    # Stoploss hit - exit position
                    signals[i] = 0.0
                    in_position = False
                    position_side = 0
                    entry_price = 0.0
                    entry_atr = 0.0
                    continue
            
            elif position_side == -1:
                stoploss_price = entry_price + ATR_STOP_MULT * entry_atr
                if close[i] > stoploss_price:
                    # Stoploss hit - exit position
                    signals[i] = 0.0
                    in_position = False
                    position_side = 0
                    entry_price = 0.0
                    entry_atr = 0.0
                    continue
            
            # Check for trend reversal exit
            if position_side == 1 and trend_signal == -1:
                signals[i] = 0.0
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                continue
            
            elif position_side == -1 and trend_signal == 1:
                signals[i] = 0.0
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                continue
            
            # Hold position
            signals[i] = float(position_side) * current_size
            continue
        
        # === Entry Logic: Need both trend and entry signals aligned ===
        if trend_signal == 1 and entry_signal == 1:
            # Enter long
            signals[i] = current_size
            in_position = True
            position_side = 1
            entry_price = close[i]
            entry_atr = atr_15m[i]
        
        elif trend_signal == -1 and entry_signal == -1:
            # Enter short
            signals[i] = -current_size
            in_position = True
            position_side = -1
            entry_price = close[i]
            entry_atr = atr_15m[i]
        
        else:
            signals[i] = 0.0
    
    return signals