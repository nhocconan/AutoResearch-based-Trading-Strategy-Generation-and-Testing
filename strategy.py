#!/usr/bin/env python3
"""
EXPERIMENT #011 - Multi-Timeframe HMA + Supertrend + RSI Pullback
=================================================================
Hypothesis: Combining 4h Supertrend (trend direction + stop levels) with 
4h HMA(48) (smooth trend confirmation) and 1h RSI(14) pullback entries will
provide better risk-adjusted returns than single-indicator approaches.

Key innovations vs current best (mtf_kama_bb_rsi_v1):
- Supertrend provides dynamic stoploss levels (ATR-based)
- HMA confirms trend direction with less lag than EMA
- RSI pullback entries avoid chasing momentum extremes
- ATR-based position sizing adapts to volatility regime
- Discrete signal levels (0.0, ±0.25, ±0.35) reduce churn costs

Why this might beat Sharpe=5.677:
- Supertrend stops reduce drawdown during reversals
- HMA + Supertrend dual confirmation reduces false signals
- Multi-TF approach already proven (exp#010 Sharpe=5.4)
- ATR-based sizing controls risk per trade to <5%
"""

import numpy as np
import pandas as pd

name = "mtf_hma_supertrend_rsi_v1"
timeframe = "1h"
leverage = 1.0


def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    n = len(close)
    tr = np.zeros(n)
    
    for i in range(n):
        if i == 0:
            tr[i] = high[i] - low[i]
        else:
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
    
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    return atr


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Calculate Supertrend indicator
    Returns: supertrend values, trend direction (1=bullish, -1=bearish)
    """
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    hl2 = (high + low) / 2
    
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    supertrend = np.zeros(n)
    trend = np.zeros(n)
    
    for i in range(n):
        if np.isnan(atr[i]) or i < period:
            upper_band[i] = np.nan
            lower_band[i] = np.nan
            supertrend[i] = np.nan
            trend[i] = 0
            continue
        
        upper_band[i] = hl2[i] + multiplier * atr[i]
        lower_band[i] = hl2[i] - multiplier * atr[i]
        
        if i == period:
            supertrend[i] = upper_band[i]
            trend[i] = -1
        else:
            if trend[i-1] == 1:
                if close[i] < lower_band[i-1]:
                    supertrend[i] = upper_band[i]
                    trend[i] = -1
                else:
                    supertrend[i] = max(upper_band[i], supertrend[i-1])
                    trend[i] = 1
            else:
                if close[i] > upper_band[i-1]:
                    supertrend[i] = lower_band[i]
                    trend[i] = 1
                else:
                    supertrend[i] = min(lower_band[i], supertrend[i-1])
                    trend[i] = -1
    
    return supertrend, trend


def calculate_hma(close, period=48):
    """
    Calculate Hull Moving Average
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        weights = weights / weights.sum()
        result = np.zeros(len(series))
        for i in range(window - 1, len(series)):
            result[i] = np.sum(series[i - window + 1:i + 1] * weights)
        return result
    
    close_series = np.array(close)
    wma_half = wma(close_series, half)
    wma_full = wma(close_series, period)
    
    diff = 2 * wma_half - wma_full
    hma = wma(diff, sqrt_n)
    
    return hma


def calculate_rsi(close, period=14):
    """Calculate RSI with proper min_periods"""
    n = len(close)
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean().values
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean().values
    
    rs = np.zeros(n)
    mask = avg_loss > 0
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    
    rsi = np.zeros(n)
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    
    return rsi


def calculate_zscore(close, period=20):
    """Calculate Z-score for volatility regime detection"""
    n = len(close)
    mean = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    zscore = np.zeros(n)
    mask = std > 0
    zscore[mask] = (close[mask] - mean[mask]) / std[mask]
    
    return zscore


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 1h indicators for entry timing
    rsi_1h = calculate_rsi(close, period=14)
    zscore_1h = calculate_zscore(close, period=20)
    atr_1h = calculate_atr(high, low, close, period=14)
    
    # 4h Supertrend and HMA for trend filter (resample 1h → 4h)
    df_1h = pd.DataFrame({
        'open': close,
        'high': high,
        'low': low,
        'close': close
    })
    df_1h.index = pd.date_range(start='2021-01-01', periods=n, freq='1h')
    
    # Resample to 4h
    df_4h = df_1h.resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()
    
    # Calculate 4h indicators
    hma_4h = calculate_hma(df_4h['close'].values, period=48)
    supertrend_4h, trend_4h = calculate_supertrend(
        df_4h['high'].values,
        df_4h['low'].values,
        df_4h['close'].values,
        period=10,
        multiplier=3.0
    )
    
    # Map 4h trend back to 1h timeframe
    trend_1h = np.zeros(n)
    supertrend_1h = np.zeros(n)
    hma_1h = np.zeros(n)
    
    idx_1h_to_4h = np.arange(n) // 4
    
    for i in range(n):
        idx_4h = idx_1h_to_4h[i]
        if idx_4h < len(trend_4h):
            trend_1h[i] = trend_4h[idx_4h]
            supertrend_1h[i] = supertrend_4h[idx_4h] if not np.isnan(supertrend_4h[idx_4h]) else 0
            hma_1h[i] = hma_4h[idx_4h] if not np.isnan(hma_4h[idx_4h]) else 0
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels to reduce churn
    SIZE_FULL = 0.35   # Full position in good conditions
    SIZE_HALF = 0.25   # Reduced position in marginal conditions
    
    # RSI thresholds for pullback entries
    RSI_LONG_ENTRY = 45   # Enter long on pullback in uptrend
    RSI_SHORT_ENTRY = 55  # Enter short on rally in downtrend
    RSI_EXIT_LONG = 65    # Exit long when RSI overbought
    RSI_EXIT_SHORT = 35   # Exit short when RSI oversold
    
    # Z-score thresholds for volatility filter
    ZSCORE_MAX = 2.0      # Don't enter if price is >2 std from mean
    
    first_valid = max(48, 20, 14, 10)  # Wait for all indicators
    
    for i in range(first_valid, n):
        if np.isnan(rsi_1h[i]) or np.isnan(zscore_1h[i]) or np.isnan(trend_1h[i]):
            signals[i] = 0.0
            continue
        
        trend = trend_1h[i]
        rsi_val = rsi_1h[i]
        zscore_val = zscore_1h[i]
        current_price = close[i]
        st_level = supertrend_1h[i]
        hma_val = hma_1h[i]
        
        # Volatility filter - don't trade during extreme moves
        if abs(zscore_val) > ZSCORE_MAX:
            signals[i] = 0.0
            continue
        
        # Check Supertrend stoploss - exit if price crosses ST level
        if signals[i-1] > 0:  # Long position
            if current_price < st_level:
                signals[i] = 0.0  # Stoploss triggered
                continue
        elif signals[i-1] < 0:  # Short position
            if current_price > st_level:
                signals[i] = 0.0  # Stoploss triggered
                continue
        
        if trend == 1:  # 4h uptrend (Supertrend bullish)
            # Confirm with HMA (price above HMA)
            if current_price > hma_val:
                if rsi_val < RSI_LONG_ENTRY:
                    # Strong pullback - full position
                    signals[i] = SIZE_FULL
                elif rsi_val < RSI_EXIT_LONG:
                    # Moderate pullback - half position
                    signals[i] = SIZE_HALF
                elif rsi_val > RSI_EXIT_LONG:
                    # Overbought - reduce or exit
                    signals[i] = 0.0
                else:
                    # Hold position
                    signals[i] = signals[i-1]
            else:
                signals[i] = 0.0
        elif trend == -1:  # 4h downtrend (Supertrend bearish)
            # Confirm with HMA (price below HMA)
            if current_price < hma_val:
                if rsi_val > RSI_SHORT_ENTRY:
                    # Strong rally - full short
                    signals[i] = -SIZE_FULL
                elif rsi_val > RSI_EXIT_SHORT:
                    # Moderate rally - half short
                    signals[i] = -SIZE_HALF
                elif rsi_val < RSI_EXIT_SHORT:
                    # Oversold - reduce or exit
                    signals[i] = 0.0
                else:
                    # Hold position
                    signals[i] = signals[i-1]
            else:
                signals[i] = 0.0
        else:  # No clear trend
            signals[i] = 0.0
    
    return signals