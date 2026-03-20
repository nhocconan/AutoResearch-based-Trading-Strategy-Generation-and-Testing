#!/usr/bin/env python3
"""
EXPERIMENT #006 - KAMA Adaptive Trend + Bollinger Mean Reversion + Volume Filter
================================================================================
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adjusts its speed based on market
efficiency ratio, making it superior to fixed MA in trending vs choppy markets.
Combined with Bollinger Band mean reversion entries and volume confirmation, this
should capture trends early while avoiding false breakouts.

Key differences from previous strategies:
- KAMA(10) adaptive trend instead of fixed EMA/HMA (responds to volatility regime)
- Bollinger Band %B for mean reversion entries (price near bands = opportunity)
- Volume confirmation filter (avoid low-volume false signals)
- ATR trailing stop with 2.5*ATR distance

Why this might beat Sharpe=2.931:
- KAMA adapts to market conditions (fast in trends, slow in chop)
- BB %B gives precise entry timing within trend
- Volume filter reduces whipsaw on low-liquidity bars
"""

import numpy as np
import pandas as pd

name = "mtf_kama_bb_volume_v1"
timeframe = "1h"
leverage = 1.0


def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average
    ER = Efficiency Ratio (signal-to-noise)
    SC = Smoothing Constant (adjusts based on ER)
    """
    n = len(close)
    kama = np.zeros(n)
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(er_period, n):
        signal = abs(close[i] - close[i - er_period])
        noise = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if noise > 0:
            er[i] = signal / noise
    
    # Calculate Smoothing Constant
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama


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


def calculate_bb_percentile(close, period=20, std_mult=2.0, lookback=100):
    """Calculate Bollinger Band Width percentile for regime detection"""
    n = len(close)
    mean = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = mean + std_mult * std
    lower = mean - std_mult * std
    bb_width = (upper - lower) / mean
    
    percentile = np.zeros(n)
    for i in range(lookback - 1, n):
        window = bb_width[i - lookback + 1:i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            percentile[i] = np.sum(valid <= bb_width[i]) / len(valid)
    
    return percentile


def calculate_bb_pct(close, period=20, std_mult=2.0):
    """Calculate Bollinger Band %B (position within bands)"""
    n = len(close)
    mean = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = mean + std_mult * std
    lower = mean - std_mult * std
    
    pct_b = np.zeros(n)
    mask = (upper - lower) > 0
    pct_b[mask] = (close[mask] - lower[mask]) / (upper[mask] - lower[mask])
    
    return pct_b


def calculate_volume_ma(volume, period=20):
    """Calculate volume moving average"""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values if "volume" in prices.columns else np.ones(len(close))
    n = len(close)
    
    # 1h indicators for entry timing and risk
    bb_pct_1h = calculate_bb_pct(close, period=20, std_mult=2.0)
    atr_1h = calculate_atr(high, low, close, period=14)
    bb_width_pct_1h = calculate_bb_percentile(close, period=20, std_mult=2.0, lookback=100)
    vol_ma_1h = calculate_volume_ma(volume, period=20)
    
    # 4h KAMA for adaptive trend filter (resample 1h → 4h)
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
    n_4h = len(c_4h)
    
    # Calculate 4h KAMA
    kama_4h = calculate_kama(c_4h, er_period=10, fast_period=2, slow_period=30)
    
    # 4h trend direction based on KAMA slope and price position
    trend_4h = np.zeros(n_4h)
    for i in range(30, n_4h):
        if kama_4h[i] > 0 and kama_4h[i-1] > 0:
            kama_slope = (kama_4h[i] - kama_4h[i-1]) / kama_4h[i-1]
            price_vs_kama = (c_4h[i] - kama_4h[i]) / kama_4h[i]
            
            # Trend up: KAMA sloping up AND price above KAMA
            if kama_slope > 0.001 and price_vs_kama > 0.005:
                trend_4h[i] = 1
            # Trend down: KAMA sloping down AND price below KAMA
            elif kama_slope < -0.001 and price_vs_kama < -0.005:
                trend_4h[i] = -1
    
    # Map 4h trend back to 1h timeframe
    trend_1h = np.zeros(n)
    idx_1h_to_4h = np.arange(n) // 4
    
    for i in range(n):
        idx_4h = min(idx_1h_to_4h[i], len(trend_4h) - 1)
        if idx_4h < len(trend_4h):
            trend_1h[i] = trend_4h[idx_4h]
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels to reduce churn
    SIZE_FULL = 0.35   # Full position in good conditions
    SIZE_HALF = 0.20   # Reduced position in marginal conditions
    
    # BB %B thresholds for mean reversion entries
    BB_PCT_LONG_ENTRY = 0.30   # Enter long when price near lower band
    BB_PCT_SHORT_ENTRY = 0.70  # Enter short when price near upper band
    BB_PCT_EXIT = 0.50         # Exit when price returns to middle
    
    # BB Width percentile thresholds for volatility filter
    BB_WIDTH_PCT_MIN = 0.15    # Don't trade in extremely low vol
    BB_WIDTH_PCT_MAX = 0.90    # Don't trade in extremely high vol
    
    # Volume filter
    VOLUME_MIN_RATIO = 0.8     # Volume must be at least 80% of average
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5
    
    first_valid = max(80, 30, 14, 100)  # Wait for all indicators
    
    # Track positions for trailing stop logic
    position_price = np.zeros(n)
    position_side = np.zeros(n)  # 1=long, -1=short, 0=none
    
    for i in range(first_valid, n):
        if np.isnan(bb_pct_1h[i]) or np.isnan(atr_1h[i]) or np.isnan(bb_width_pct_1h[i]):
            signals[i] = 0.0
            continue
        
        trend = trend_1h[i]
        bb_pct = bb_pct_1h[i]
        bb_width_pct = bb_width_pct_1h[i]
        atr = atr_1h[i]
        price = close[i]
        vol_ratio = volume[i] / vol_ma_1h[i] if vol_ma_1h[i] > 0 else 0
        
        # Volatility filter - avoid extreme regimes
        if bb_width_pct < BB_WIDTH_PCT_MIN or bb_width_pct > BB_WIDTH_PCT_MAX:
            signals[i] = 0.0
            position_price[i] = 0
            position_side[i] = 0
            continue
        
        # Volume filter - avoid low-volume bars
        if vol_ratio < VOLUME_MIN_RATIO:
            signals[i] = 0.0
            position_price[i] = 0
            position_side[i] = 0
            continue
        
        # ATR filter - avoid trading when ATR is extremely high
        if atr > 0 and atr / price > 0.05:  # ATR > 5% of price = too volatile
            signals[i] = 0.0
            position_price[i] = 0
            position_side[i] = 0
            continue
        
        # Check trailing stop for existing positions
        if position_side[i-1] != 0 and i > 0:
            entry_price = position_price[i-1]
            side = position_side[i-1]
            
            if side == 1:  # Long position
                stoploss_price = entry_price - ATR_STOP_MULT * atr
                take_profit_price = entry_price + 2 * ATR_STOP_MULT * atr
                
                if price < stoploss_price:
                    # Stoploss triggered
                    signals[i] = 0.0
                    position_price[i] = 0
                    position_side[i] = 0
                    continue
                elif price > take_profit_price:
                    # Take profit - reduce to half
                    signals[i] = SIZE_HALF
                    position_price[i] = price
                    position_side[i] = 1
                    continue
                elif bb_pct > BB_PCT_EXIT:
                    # Mean reversion complete - exit
                    signals[i] = 0.0
                    position_price[i] = 0
                    position_side[i] = 0
                    continue
                else:
                    # Hold position
                    signals[i] = signals[i-1]
                    position_price[i] = entry_price
                    position_side[i] = side
                    continue
            
            elif side == -1:  # Short position
                stoploss_price = entry_price + ATR_STOP_MULT * atr
                take_profit_price = entry_price - 2 * ATR_STOP_MULT * atr
                
                if price > stoploss_price:
                    # Stoploss triggered
                    signals[i] = 0.0
                    position_price[i] = 0
                    position_side[i] = 0
                    continue
                elif price < take_profit_price:
                    # Take profit - reduce to half
                    signals[i] = -SIZE_HALF
                    position_price[i] = price
                    position_side[i] = -1
                    continue
                elif bb_pct < BB_PCT_EXIT:
                    # Mean reversion complete - exit
                    signals[i] = 0.0
                    position_price[i] = 0
                    position_side[i] = 0
                    continue
                else:
                    # Hold position
                    signals[i] = signals[i-1]
                    position_price[i] = entry_price
                    position_side[i] = side
                    continue
        
        # New entry logic
        if trend == 1:  # 4h uptrend - look for long entries
            if bb_pct < BB_PCT_LONG_ENTRY:
                # Price near lower band in uptrend = buy opportunity
                signals[i] = SIZE_FULL
                position_price[i] = price
                position_side[i] = 1
            else:
                signals[i] = 0.0
                position_price[i] = 0
                position_side[i] = 0
                
        elif trend == -1:  # 4h downtrend - look for short entries
            if bb_pct > BB_PCT_SHORT_ENTRY:
                # Price near upper band in downtrend = sell opportunity
                signals[i] = -SIZE_FULL
                position_price[i] = price
                position_side[i] = -1
            else:
                signals[i] = 0.0
                position_price[i] = 0
                position_side[i] = 0
        else:  # No clear trend
            signals[i] = 0.0
            position_price[i] = 0
            position_side[i] = 0
    
    return signals