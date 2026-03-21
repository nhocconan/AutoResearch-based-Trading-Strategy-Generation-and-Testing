#!/usr/bin/env python3
"""
EXPERIMENT #001 - 30m KAMA-HMA Trend Following with RSI Pullback
==================================================================================================
Hypothesis: Current best (mtf_hma_rsi_zscore_v1, Sharpe=5.4) uses 4h HMA + 1h RSI. 
This version tests 30m primary with 4h HTF filter - should reduce noise vs 15m while 
maintaining better trade frequency than 1h. Key changes:

1. Primary=30m (sweeter spot between 15m noise and 1h slowness)
2. 4h HMA for major trend (proven in current best)
3. 30m KAMA for adaptive trend confirmation (better than EMA in ranges)
4. RSI pullback entries (40-60 zone for trend continuation)
5. Z-score filter to avoid overextended entries
6. Simplified stoploss: 2*ATR hard stop, no complex trailing

Why this should work:
- 30m has fewer false signals than 15m (less noise)
- More trades than 1h (better statistics)
- KAMA adapts to volatility better than HMA alone
- RSI pullback in trend direction = high probability entries
- Conservative sizing (0.25-0.35) controls drawdown
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_kama_hma_rsi_zscore_30m_4h_v1"
timeframe = "30m"
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
    """
    Hull Moving Average - reduces lag while maintaining smoothness
    HMA = WMA(2*WMA(n/2) - WMA(n))
    """
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    half_period = period // 2
    
    # WMA helper
    def wma(series, window):
        weights = np.arange(1, window + 1)
        weights = weights / weights.sum()
        return pd.Series(series).rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights), raw=True
        ).values
    
    wma_half = wma(close, half_period)
    wma_full = wma(close, period)
    
    # Combine
    combined = 2 * wma_half - wma_full
    hma = wma(combined, int(np.sqrt(period)))
    
    # Handle NaN
    hma = np.nan_to_num(hma, nan=0.0)
    return hma


def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts to market noise - moves fast in trends, slow in ranges
    """
    n = len(close)
    if n < er_period + slow_period:
        return np.zeros(n)
    
    # Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(er_period, n):
        price_change = abs(close[i] - close[i - er_period])
        volatility = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if volatility > 0:
            er[i] = price_change / volatility
        else:
            er[i] = 0
    
    # Smoothing constants
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    
    kama = np.zeros(n)
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
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
    
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    rs = np.zeros(n)
    mask = avg_loss > 0
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rs[~mask] = 100
    
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_zscore(close, period=20):
    """Calculate Z-score for overextension detection"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    rolling_mean = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    rolling_std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    zscore = np.zeros(n)
    mask = rolling_std > 0
    zscore[mask] = (close[mask] - rolling_mean[mask]) / rolling_std[mask]
    
    return zscore


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # ========== 30m INDICATORS (ENTRY TIMING) ==========
    atr_30m = calculate_atr(high, low, close, period=14)
    rsi_30m = calculate_rsi(close, period=14)
    kama_30m = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    hma_30m = calculate_hma(close, period=21)
    zscore_30m = calculate_zscore(close, period=20)
    
    # ========== 4h INDICATORS (TREND FILTER) - PROPER MTF ==========
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        
        # 4h HMA for major trend direction
        hma_4h = calculate_hma(close_4h, period=21)
        
        # Align to 30m timeframe (auto shift for completed bars)
        hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
        
    except Exception:
        hma_4h_aligned = np.zeros(n)
    
    # ========== SIGNAL GENERATION ==========
    signals = np.zeros(n)
    
    # Position sizing - CONSERVATIVE
    SIZE_BASE = 0.25    # Standard position
    SIZE_HIGH = 0.35    # High conviction (all filters agree)
    MAX_SIZE = 0.40     # Absolute maximum
    
    # Thresholds
    RSI_LONG_MIN = 40   # Pullback entry zone
    RSI_LONG_MAX = 60
    RSI_SHORT_MIN = 40
    RSI_SHORT_MAX = 60
    
    ZSCORE_EXTREME = 2.0  # Don't enter if Z-score > 2 (overextended)
    
    # ATR stoploss
    ATR_STOP_MULT = 2.0
    
    first_valid = max(100, 50)
    
    # Track position state
    position_side = np.zeros(n, dtype=int)
    entry_price = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        # Skip invalid data
        if np.isnan(atr_30m[i]) or atr_30m[i] == 0 or np.isnan(rsi_30m[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_30m[i]
        rsi_val = rsi_30m[i]
        zscore_val = zscore_30m[i]
        kama_val = kama_30m[i]
        hma_30m_val = hma_30m[i]
        
        # 4h trend filter
        hma_4h_val = hma_4h_aligned[i]
        
        # ========== CHECK EXISTING POSITIONS ==========
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_high = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
            prev_low = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
            
            # Update highest/lowest since entry
            if prev_side == 1:
                current_high = max(prev_high, price)
                current_low = prev_low
            else:
                current_high = prev_high
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
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
            
            # Hold position if no exit triggered
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
            continue
        
        # ========== TREND DIRECTION (4h HMA) ==========
        # 4h HMA slope determines major trend
        trend_direction = 0
        if hma_4h_val > 0:
            # Compare price to 4h HMA
            if price > hma_4h_val * 1.002:  # 0.2% above
                trend_direction = 1
            elif price < hma_4h_val * 0.998:  # 0.2% below
                trend_direction = -1
        
        # ========== LOCAL TREND CONFIRMATION (30m KAMA vs HMA) ==========
        local_trend = 0
        if kama_val > hma_30m_val * 1.001:
            local_trend = 1
        elif kama_val < hma_30m_val * 0.999:
            local_trend = -1
        
        # ========== ENTRY SIGNALS ==========
        # Long setup: 4h uptrend + 30m pullback (RSI 40-60) + not overextended
        long_signal = False
        if trend_direction == 1 and local_trend >= 0:
            if RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX:
                if abs(zscore_val) < ZSCORE_EXTREME:
                    long_signal = True
        
        # Short setup: 4h downtrend + 30m pullback (RSI 40-60) + not overextended
        short_signal = False
        if trend_direction == -1 and local_trend <= 0:
            if RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX:
                if abs(zscore_val) < ZSCORE_EXTREME:
                    short_signal = True
        
        # ========== POSITION SIZING ==========
        if long_signal:
            # High conviction: all filters agree
            if local_trend == 1 and rsi_val < 55:
                size = SIZE_HIGH
            else:
                size = SIZE_BASE
            
            signals[i] = size
            position_side[i] = 1
            entry_price[i] = price
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
            
        elif short_signal:
            # High conviction: all filters agree
            if local_trend == -1 and rsi_val > 45:
                size = SIZE_HIGH
            else:
                size = SIZE_BASE
            
            signals[i] = -size
            position_side[i] = -1
            entry_price[i] = price
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        
        # Track state for existing positions
        if position_side[i] != 0 and entry_price[i] == 0:
            entry_price[i] = entry_price[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
    
    return signals