#!/usr/bin/env python3
"""
EXPERIMENT #012 - KAMA Adaptive Trend + BB Squeeze Breakout + RSI Filter
=========================================================================
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market efficiency,
providing better trend signals than fixed MAs. Combined with Bollinger Band
squeeze breakouts (volatility expansion) and RSI momentum filter, this should
capture strong trends while avoiding choppy periods.

Key differences from mtf_donchian_rsi_atr_v1:
- KAMA(10) instead of Donchian for adaptive trend following
- BB Squeeze detection (BB width < 20th percentile) for breakout entries
- RSI momentum confirmation (RSI > 55 for long, < 45 for short)
- ATR trailing stop at 2.0*ATR distance

Why this might beat Sharpe=5.677:
- KAMA reduces whipsaw in choppy markets (ER-based adaptation)
- BB squeeze captures volatility expansion breakouts
- RSI filter ensures momentum confirmation before entry
"""

import numpy as np
import pandas as pd

name = "mtf_kama_bbsqueeze_rsi_v1"
timeframe = "1h"
leverage = 1.0


def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average
    ER = |close - close[n]| / sum(|close[i] - close[i-1]|)
    SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
    KAMA = KAMA_prev + SC * (close - KAMA_prev)
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
    
    # Smoothing constants
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    
    # Calculate KAMA
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


def calculate_bb_width(close, period=20, std_mult=2.0):
    """Calculate Bollinger Band Width"""
    n = len(close)
    mean = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = mean + std_mult * std
    lower = mean - std_mult * std
    bb_width = (upper - lower) / mean
    
    return bb_width, upper, lower


def calculate_bb_percentile(bb_width, lookback=100):
    """Calculate BB Width percentile for squeeze detection"""
    n = len(bb_width)
    percentile = np.zeros(n)
    
    for i in range(lookback - 1, n):
        window = bb_width[i - lookback + 1:i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            percentile[i] = np.sum(valid <= bb_width[i]) / len(valid)
    
    return percentile


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 1h indicators for entry timing and risk
    rsi_1h = calculate_rsi(close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    bb_width_1h, bb_upper_1h, bb_lower_1h = calculate_bb_width(close, period=20, std_mult=2.0)
    bb_pct_1h = calculate_bb_percentile(bb_width_1h, lookback=100)
    
    # 4h KAMA for adaptive trend (resample 1h → 4h)
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
    
    c_4h = df_4h['close'].values
    
    # Calculate 4h KAMA
    kama_4h = calculate_kama(c_4h, er_period=10, fast_period=2, slow_period=30)
    
    # 4h trend direction based on KAMA slope and price position
    trend_4h = np.zeros(len(c_4h))
    for i in range(20, len(c_4h)):
        if kama_4h[i] > 0 and kama_4h[i-1] > 0:
            kama_slope = kama_4h[i] - kama_4h[i - 5]  # 5-period slope
            price_vs_kama = (c_4h[i] - kama_4h[i]) / kama_4h[i]
            
            if kama_slope > 0 and price_vs_kama > 0.005:
                trend_4h[i] = 1  # Bullish
            elif kama_slope < 0 and price_vs_kama < -0.005:
                trend_4h[i] = -1  # Bearish
    
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
    SIZE_HALF = 0.20   # Reduced position in marginal conditions
    
    # BB Squeeze thresholds
    BB_SQUEEZE_PCT = 0.25  # BB width in bottom 25% = squeeze
    BB_EXPANSION_PCT = 0.60  # BB width expanding above 60th percentile
    
    # RSI thresholds for momentum confirmation
    RSI_LONG_CONFIRM = 55   # RSI must be > 55 for long confirmation
    RSI_SHORT_CONFIRM = 45  # RSI must be < 45 for short confirmation
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    
    first_valid = max(80, 20, 14, 100)  # Wait for all indicators
    
    # Track entry prices for trailing stop
    entry_price_long = np.zeros(n)
    entry_price_short = np.zeros(n)
    in_position_long = np.zeros(n, dtype=bool)
    in_position_short = np.zeros(n, dtype=bool)
    
    for i in range(first_valid, n):
        if np.isnan(rsi_1h[i]) or np.isnan(atr_1h[i]) or np.isnan(bb_pct_1h[i]) or np.isnan(trend_1h[i]):
            signals[i] = 0.0
            continue
        
        trend = trend_1h[i]
        rsi_val = rsi_1h[i]
        bb_pct = bb_pct_1h[i]
        atr = atr_1h[i]
        price = close[i]
        
        # ATR filter - avoid trading when ATR is extremely high
        if atr > 0 and atr / price > 0.06:  # ATR > 6% of price = too volatile
            signals[i] = 0.0
            in_position_long[i] = False
            in_position_short[i] = False
            continue
        
        # Check trailing stop for existing positions first
        if in_position_long[i-1] if i > 0 else False:
            stoploss_price = entry_price_long[i-1] if i > 0 else price - ATR_STOP_MULT * atr
            if i > 0:
                stoploss_price = entry_price_long[max(0, i-1)] - ATR_STOP_MULT * atr
            if price < stoploss_price:
                signals[i] = 0.0
                in_position_long[i] = False
                continue
            else:
                # Hold position, check for take profit reduction
                profit_ratio = (price - entry_price_long[max(0, i-1)]) / (ATR_STOP_MULT * atr) if atr > 0 else 0
                if profit_ratio >= 2.0:
                    signals[i] = SIZE_HALF  # Reduce to half at 2R profit
                else:
                    signals[i] = SIZE_FULL
                in_position_long[i] = True
                continue
        
        if in_position_short[i-1] if i > 0 else False:
            stoploss_price = entry_price_short[max(0, i-1)] + ATR_STOP_MULT * atr if i > 0 else price + ATR_STOP_MULT * atr
            if price > stoploss_price:
                signals[i] = 0.0
                in_position_short[i] = False
                continue
            else:
                # Hold position, check for take profit reduction
                profit_ratio = (entry_price_short[max(0, i-1)] - price) / (ATR_STOP_MULT * atr) if atr > 0 else 0
                if profit_ratio >= 2.0:
                    signals[i] = -SIZE_HALF  # Reduce to half at 2R profit
                else:
                    signals[i] = -SIZE_FULL
                in_position_short[i] = True
                continue
        
        # New entry logic
        if trend == 1:  # 4h uptrend
            # BB squeeze breakout + RSI confirmation
            if bb_pct < BB_SQUEEZE_PCT or bb_pct > BB_EXPANSION_PCT:
                if rsi_val > RSI_LONG_CONFIRM:
                    signals[i] = SIZE_FULL
                    entry_price_long[i] = price
                    in_position_long[i] = True
                elif rsi_val > 50:
                    signals[i] = SIZE_HALF
                    entry_price_long[i] = price
                    in_position_long[i] = True
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        elif trend == -1:  # 4h downtrend
            # BB squeeze breakout + RSI confirmation
            if bb_pct < BB_SQUEEZE_PCT or bb_pct > BB_EXPANSION_PCT:
                if rsi_val < RSI_SHORT_CONFIRM:
                    signals[i] = -SIZE_FULL
                    entry_price_short[i] = price
                    in_position_short[i] = True
                elif rsi_val < 50:
                    signals[i] = -SIZE_HALF
                    entry_price_short[i] = price
                    in_position_short[i] = True
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        else:  # No clear trend
            signals[i] = 0.0
    
    return signals