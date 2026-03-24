#!/usr/bin/env python3
"""
Experiment #1589: 4h Primary + 1d HTF — Fisher Transform Reversion Strategy

Hypothesis: After 11 failed 4h experiments with complex regime switching (Choppiness, CRSI, dual-regime),
simpler reversal detection works better in bear/range markets. The Ehlers Fisher Transform normalizes
price to -1/+1 range and provides clear reversal signals that work well in 2022 crash and 2025 bear.

Key innovations:
1. Fisher Transform(9) on 4h - catches reversals better than RSI in bear markets
2. 1d HMA(21) for trend bias (proven in mtf_1d_donchian_hma_rsi_1w_atr_v1 - Sharpe 0.618)
3. BB(20,2.0) width filter - only trade when volatility expands (avoids chop)
4. Volume confirmation - taker_buy_volume ratio > 0.55 for longs, < 0.45 for shorts
5. ATR(14) 2.5x trailing stop for drawdown control
6. Discrete position sizing (0.28) to minimize fee churn

Why this should beat Sharpe 0.618:
- Fisher Transform excels in bear/range markets (2025 test period is bearish)
- Simpler logic = more reliable signals = more trades (>30/year target)
- Volume filter reduces false signals (major issue in prior 4h experiments)
- BB width filter avoids trading in low-vol chop (whipsaw killer)
- 4h targets 20-50 trades/year — optimal for fee efficiency

Timeframe: 4h (required for this experiment)
HTF: 1d HMA for bias (use mtf_data helper - call ONCE before loop)
Target: Sharpe > 0.618, trades > 10/symbol train, > 3/symbol test, DD > -50%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_reversion_1d_hma_bb_vol_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_fisher(close, period=9):
    """
    Ehlers Fisher Transform
    Normalizes price to -1 to +1 range for clear reversal signals
    Fisher = 0.5 * ln((1 + X) / (1 - X)) where X = 0.33 * 2 * ((close - low_min) / (high_max - low_min) - 0.5)
    
    Long signal: Fisher crosses above -1.5 (oversold reversal)
    Short signal: Fisher crosses below +1.5 (overbought reversal)
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.full(n, np.nan)
    fisher_signal = np.full(n, np.nan)  # previous bar fisher for crossover detection
    
    # Calculate highest high and lowest low over period
    high_max = np.full(n, np.nan)
    low_min = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        high_max[i] = np.max(close[i - period + 1:i + 1])
        low_min[i] = np.min(close[i - period + 1:i + 1])
    
    # Calculate X value (normalized price)
    X = np.full(n, np.nan)
    for i in range(period - 1, n):
        if high_max[i] > low_min[i]:
            X[i] = 0.66 * ((close[i] - low_min[i]) / (high_max[i] - low_min[i]) - 0.5)
            # Clamp X to avoid division by zero in log
            X[i] = max(-0.999, min(0.999, X[i]))
    
    # Calculate Fisher Transform
    for i in range(period - 1, n):
        if not np.isnan(X[i]):
            fisher[i] = 0.5 * np.log((1 + X[i]) / (1 - X[i]))
    
    # Store previous fisher for crossover detection
    fisher_signal[1:] = fisher[:-1]
    
    return fisher, fisher_signal

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(period // 2, 1)
    sqrt_period = max(int(np.sqrt(period)), 1)
    
    def wma(data, w_period):
        if w_period < 1:
            return np.full(len(data), np.nan)
        result = np.full(len(data), np.nan)
        weights = np.arange(1, w_period + 1, dtype=float)
        weights /= weights.sum()
        for i in range(w_period - 1, len(data)):
            window = data[i - w_period + 1:i + 1]
            if np.any(np.isnan(window)):
                continue
            result[i] = np.sum(window * weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan)
    mask = ~np.isnan(wma_half) & ~np.isnan(wma_full)
    diff[mask] = 2.0 * wma_half[mask] - wma_full[mask]
    
    hma = wma(diff, sqrt_period)
    return hma

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands with proper min_periods"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    bandwidth = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
        if np.any(np.isnan(window)):
            continue
        sma[i] = np.mean(window)
        std = np.std(window, ddof=0)
        upper[i] = sma[i] + std_mult * std
        lower[i] = sma[i] - std_mult * std
        if sma[i] > 1e-10:
            bandwidth[i] = (upper[i] - lower[i]) / sma[i]
    
    return upper, lower, bandwidth

def calculate_atr(high, low, close, period=14):
    """Average True Range with proper min_periods"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_ratio(taker_buy_volume, volume):
    """Taker buy volume ratio (0-1 scale)"""
    n = len(volume)
    ratio = np.full(n, np.nan)
    mask = volume > 1e-10
    ratio[mask] = taker_buy_volume[mask] / volume[mask]
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_volume = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    atr = calculate_atr(high, low, close, period=14)
    fisher, fisher_signal = calculate_fisher(close, period=9)
    bb_upper, bb_lower, bb_bandwidth = calculate_bollinger(close, period=20, std_mult=2.0)
    vol_ratio = calculate_volume_ratio(taker_buy_volume, volume)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # Track BB bandwidth percentile for volatility regime (simplified)
    bb_width_ma = pd.Series(bb_bandwidth).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(bb_bandwidth[i]) or np.isnan(bb_width_ma[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND BIAS (1d HMA) ===
        daily_bull = close[i] > hma_1d_aligned[i]
        daily_bear = close[i] < hma_1d_aligned[i]
        
        # === VOLATILITY REGIME (BB Bandwidth) ===
        # Only trade when volatility is expanding (above 50-bar MA of bandwidth)
        vol_expanding = bb_bandwidth[i] > bb_width_ma[i]
        
        # === FISHER REVERSAL SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        fisher_long_cross = (fisher_signal[i] < -1.5) and (fisher[i] >= -1.5)
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_short_cross = (fisher_signal[i] > 1.5) and (fisher[i] <= 1.5)
        
        # === VOLUME CONFIRMATION ===
        vol_bull = vol_ratio[i] > 0.55  # More buying pressure
        vol_bear = vol_ratio[i] < 0.45  # More selling pressure
        
        # === PRIMARY SIGNAL ===
        desired_signal = 0.0
        
        # LONG: Fisher long cross + Daily bull + Vol expanding + Volume confirmation
        if fisher_long_cross and daily_bull and vol_expanding and vol_bull:
            desired_signal = BASE_SIZE
        
        # SHORT: Fisher short cross + Daily bear + Vol expanding + Volume confirmation
        elif fisher_short_cross and daily_bear and vol_expanding and vol_bear:
            desired_signal = -BASE_SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= BASE_SIZE * 0.85:
            final_signal = BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.85:
            final_signal = -BASE_SIZE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals