#!/usr/bin/env python3
"""
EXPERIMENT #017 - Donchian Breakout + HMA Trend + RSI Pullback + Z-Score Filter
===============================================================================
Hypothesis: Combining Donchian breakout trend (4h) with HMA confirmation (1h) 
and RSI pullback entries should capture momentum earlier than pure MA strategies.
Z-score filter avoids entering at extreme deviations (mean reversion risk).
ATR trailing stop provides dynamic risk management.

Key innovations vs mtf_keltner_rsi_adx_v1:
- Donchian(20) breakout instead of Keltner for pure price action trend
- HMA(21) for smoother trend confirmation with less lag than EMA
- Z-score(20) filter to avoid entering at >2 std dev from mean
- Discrete position sizing (0.0, ±0.25, ±0.35) to reduce churn costs
- ATR trailing stop at 2.5*ATR with proper entry price tracking

Why this might beat Sharpe=4.452:
- Donchian captures breakouts earlier than volatility-based channels
- HMA reduces whipsaw in choppy markets
- Z-score filter prevents chasing extended moves
- Multi-timeframe logic proven in mtf_hma_rsi_zscore_v1 (Sharpe=5.4)
"""

import numpy as np
import pandas as pd

name = "mtf_donchian_hma_rsi_zscore_v1"
timeframe = "1h"
leverage = 1.0


def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA calculations
    def wma(data, w):
        result = np.zeros(len(data))
        for i in range(w - 1, len(data)):
            weights = np.arange(1, w + 1)
            result[i] = np.sum(data[i - w + 1:i + 1] * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    hma = np.zeros(n)
    for i in range(sqrt_period - 1, n):
        hma[i] = wma(2 * wma_half - wma_full, sqrt_period)[i]
    
    return hma


def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel - pure price action breakout"""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower


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
    """Calculate Z-score for mean reversion filter"""
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
    
    # 1h indicators for entry timing and risk
    rsi_1h = calculate_rsi(close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    zscore_1h = calculate_zscore(close, period=20)
    hma_1h = calculate_hma(close, period=21)
    
    # 4h Donchian for trend filter (resample 1h → 4h)
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
    
    h_4h = df_4h['high'].values
    l_4h = df_4h['low'].values
    c_4h = df_4h['close'].values
    
    # Calculate 4h Donchian
    donchian_upper, donchian_lower = calculate_donchian(h_4h, l_4h, period=20)
    
    # 4h trend direction based on Donchian position
    trend_4h = np.zeros(len(c_4h))
    for i in range(20, len(c_4h)):
        channel_range = donchian_upper[i] - donchian_lower[i]
        if channel_range > 0:
            price_position = (c_4h[i] - donchian_lower[i]) / channel_range
            if price_position > 0.65:
                trend_4h[i] = 1  # Bullish (price in upper 35% of channel)
            elif price_position < 0.35:
                trend_4h[i] = -1  # Bearish (price in lower 35% of channel)
    
    # Map 4h trend back to 1h timeframe
    trend_1h = np.zeros(n)
    idx_1h_to_4h = np.arange(n) // 4
    
    for i in range(n):
        idx_4h = idx_1h_to_4h[i]
        if idx_4h < len(trend_4h):
            trend_1h[i] = trend_4h[idx_4h]
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels to reduce churn
    SIZE_FULL = 0.35   # Full position in good conditions
    SIZE_HALF = 0.25   # Reduced position in marginal conditions
    
    # RSI thresholds for pullback entries
    RSI_LONG_ENTRY = 45   # Enter long on pullback in uptrend
    RSI_SHORT_ENTRY = 55  # Enter short on rally in downtrend
    
    # Z-score thresholds for regime filter
    ZSCORE_MAX = 2.0      # Don't enter if price > 2 std dev from mean
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5
    
    first_valid = max(80, 20, 14, 21)  # Wait for all indicators
    
    # Track entry prices for trailing stop logic
    entry_price = np.zeros(n)
    position_side = np.zeros(n)  # 1 for long, -1 for short, 0 for flat
    
    for i in range(first_valid, n):
        if np.isnan(rsi_1h[i]) or np.isnan(atr_1h[i]) or np.isnan(zscore_1h[i]):
            signals[i] = 0.0
            continue
        
        trend = trend_1h[i]
        rsi_val = rsi_1h[i]
        zscore_val = zscore_1h[i]
        atr = atr_1h[i]
        price = close[i]
        hma_val = hma_1h[i]
        
        # ATR filter - avoid trading when ATR is extremely high
        if atr > 0 and atr / price > 0.05:  # ATR > 5% of price = too volatile
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Check trailing stop for existing positions
        if i > 0 and position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else price
            
            if prev_side == 1:  # Long position
                stoploss_price = prev_entry - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    continue
            elif prev_side == -1:  # Short position
                stoploss_price = prev_entry + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    continue
        
        # Z-score filter - avoid entering at extreme deviations
        if abs(zscore_val) > ZSCORE_MAX:
            # If we have a position, hold it; otherwise stay flat
            if i > 0 and position_side[i - 1] != 0:
                signals[i] = signals[i - 1]
                position_side[i] = position_side[i - 1]
                entry_price[i] = entry_price[i - 1]
            else:
                signals[i] = 0.0
                position_side[i] = 0
            continue
        
        if trend == 1:  # 4h uptrend
            # HMA confirmation - price above HMA
            hma_confirmed = price > hma_val
            
            if rsi_val < RSI_LONG_ENTRY and hma_confirmed:
                # Pullback entry - full position
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = price
            elif rsi_val < 50 and hma_confirmed:
                # Moderate pullback - half position
                signals[i] = SIZE_HALF
                position_side[i] = 1
                entry_price[i] = price
            else:
                # Hold or exit
                if i > 0 and position_side[i - 1] == 1:
                    signals[i] = signals[i - 1]
                    position_side[i] = 1
                    entry_price[i] = entry_price[i - 1]
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    
        elif trend == -1:  # 4h downtrend
            # HMA confirmation - price below HMA
            hma_confirmed = price < hma_val
            
            if rsi_val > RSI_SHORT_ENTRY and hma_confirmed:
                # Rally entry - full short
                signals[i] = -SIZE_FULL
                position_side[i] = -1
                entry_price[i] = price
            elif rsi_val > 50 and hma_confirmed:
                # Moderate rally - half short
                signals[i] = -SIZE_HALF
                position_side[i] = -1
                entry_price[i] = price
            else:
                # Hold or exit
                if i > 0 and position_side[i - 1] == -1:
                    signals[i] = signals[i - 1]
                    position_side[i] = -1
                    entry_price[i] = entry_price[i - 1]
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
        else:  # No clear trend
            signals[i] = 0.0
            position_side[i] = 0
            entry_price[i] = 0
    
    return signals