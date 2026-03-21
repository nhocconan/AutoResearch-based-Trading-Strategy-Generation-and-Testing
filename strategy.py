#!/usr/bin/env python3
"""
EXPERIMENT #003 - HMA Trend + MACD Histogram + Volume Confirmation
====================================================================================
Hypothesis: Return to HMA trend (proven in baseline Sharpe=5.4) but combine with
MACD histogram momentum entries instead of RSI pullbacks. Add volume confirmation
to ensure breakouts have conviction. This differs from #002 which used Donchian+KAMA.

Key differences from #002:
- HMA(16/48) crossover for 4h trend instead of Donchian breakouts
- MACD(12,26,9) histogram cross for 15m entry timing instead of RSI pullback
- Volume spike filter (volume > 1.5x 20-period average) for conviction
- ADX(14) > 20 filter for trend strength (same as #002)
- Z-score regime filter to avoid extreme deviations

Why this might beat Sharpe=1.442:
- HMA has less lag than Donchian for trend detection (proven in baseline)
- MACD histogram captures momentum shifts earlier than RSI extremes
- Volume confirmation filters false breakouts (major improvement)
- Different signal combination may capture regimes that Donchian misses
- HMA crossover is smoother than Donchian breakout whipsaws
"""

import numpy as np
import pandas as pd

name = "mtf_hma_macd_volume_v1"
timeframe = "15m"
leverage = 1.0


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    # WMA of period/2
    wma_half = pd.Series(close).ewm(span=period // 2, adjust=False).mean().values
    
    # WMA of period
    wma_full = pd.Series(close).ewm(span=period, adjust=False).mean().values
    
    # 2*WMA_half - WMA_full
    raw_hma = 2 * wma_half - wma_full
    
    # WMA of sqrt(period)
    hma = pd.Series(raw_hma).ewm(span=int(np.sqrt(period)), adjust=False).mean().values
    
    return hma


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD with histogram"""
    n = len(close)
    
    ema_fast = pd.Series(close).ewm(span=fast, adjust=False).mean().values
    ema_slow = pd.Series(close).ewm(span=slow, adjust=False).mean().values
    
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=signal, adjust=False).mean().values
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


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


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    n = len(close)
    adx = np.zeros(n)
    
    if n < period * 2:
        return adx
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        if high[i] - high[i - 1] > low[i - 1] - low[i]:
            plus_dm[i] = max(0, high[i] - high[i - 1])
        
        if low[i - 1] - low[i] > high[i] - high[i - 1]:
            minus_dm[i] = max(0, low[i - 1] - low[i])
    
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=period, min_periods=period).sum().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=period, min_periods=period).sum().values
    tr_smooth = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    mask = tr_smooth > 0
    plus_di[mask] = 100 * plus_dm_smooth[mask] / tr_smooth[mask]
    minus_di[mask] = 100 * minus_dm_smooth[mask] / tr_smooth[mask]
    
    dx = np.zeros(n)
    mask2 = (plus_di + minus_di) > 0
    dx[mask2] = 100 * np.abs(plus_di[mask2] - minus_di[mask2]) / (plus_di[mask2] + minus_di[mask2])
    
    adx[period * 2 - 1:] = pd.Series(dx).rolling(window=period, min_periods=period).mean().values[period * 2 - 1:]
    
    return adx


def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion filter"""
    n = len(close)
    mean = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    zscore = np.zeros(n)
    mask = std > 0
    zscore[mask] = (close[mask] - mean[mask]) / std[mask]
    
    return zscore


def calculate_volume_spike(volume, period=20, threshold=1.5):
    """Detect volume spikes above average"""
    n = len(volume)
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_spike = np.zeros(n)
    
    mask = vol_avg > 0
    vol_spike[mask] = (volume[mask] / vol_avg[mask]) > threshold
    
    return vol_spike


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values if "volume" in prices.columns else np.ones(len(close))
    n = len(close)
    
    # 15m indicators for entry timing
    macd_15m, signal_15m, hist_15m = calculate_macd(close, fast=12, slow=26, signal=9)
    atr_15m = calculate_atr(high, low, close, period=14)
    zscore_15m = calculate_zscore(close, period=20)
    adx_15m = calculate_adx(high, low, close, period=14)
    vol_spike_15m = calculate_volume_spike(volume, period=20, threshold=1.5)
    
    # 4h trend via HMA crossover (resample 15m → 4h)
    df_15m = pd.DataFrame({
        'open': close,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    df_15m.index = pd.date_range(start='2021-01-01', periods=n, freq='15min')
    
    # Resample to 4h
    df_4h = df_15m.resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    c_4h = df_4h['close'].values
    n_4h = len(c_4h)
    
    # Calculate 4h HMA crossover for trend
    hma_fast_4h = calculate_hma(c_4h, period=16)
    hma_slow_4h = calculate_hma(c_4h, period=48)
    
    # 4h trend direction based on HMA crossover
    trend_4h = np.zeros(n_4h)
    for i in range(48, n_4h):
        if hma_fast_4h[i] > hma_slow_4h[i]:
            trend_4h[i] = 1  # Bullish
        elif hma_fast_4h[i] < hma_slow_4h[i]:
            trend_4h[i] = -1  # Bearish
        else:
            trend_4h[i] = trend_4h[i - 1] if i > 0 else 0
    
    # Map 4h trend back to 15m timeframe (16 x 15m = 4h)
    trend_15m = np.zeros(n)
    idx_15m_to_4h = np.arange(n) // 16
    
    for i in range(n):
        idx_4h = idx_15m_to_4h[i]
        if idx_4h < len(trend_4h):
            trend_15m[i] = trend_4h[idx_4h]
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels to reduce churn
    SIZE_FULL = 0.35   # Full position
    SIZE_HALF = 0.20   # Half position (after take profit)
    SIZE_MIN = 0.15    # Minimum position
    
    # MACD histogram thresholds for entries
    MACD_LONG_THRESHOLD = 0.0   # Histogram crosses above zero
    MACD_SHORT_THRESHOLD = 0.0  # Histogram crosses below zero
    
    # ADX threshold for trend strength
    ADX_MIN = 20  # Only trade if ADX > 20 (strong trend)
    
    # Z-score thresholds for regime filter
    ZSCORE_MAX = 2.5  # Don't enter if price > 2.5 std dev from mean
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    
    # Take profit multiplier (2R)
    TP_MULT = 2.0
    
    # ATR volatility target for dynamic sizing
    TARGET_ATR_PCT = 0.02  # Target 2% ATR
    
    first_valid = max(80, 48, 14, 20, 28)  # Wait for all indicators
    
    # Track entry prices for stoploss/takeprofit logic
    entry_price = np.zeros(n)
    position_side = np.zeros(n)  # 1 for long, -1 for short, 0 for flat
    tp_triggered = np.zeros(n)  # Track if take profit was hit
    highest_since_entry = np.zeros(n)  # For trailing stop
    lowest_since_entry = np.zeros(n)  # For trailing stop
    
    for i in range(first_valid, n):
        if np.isnan(macd_15m[i]) or np.isnan(hist_15m[i]) or np.isnan(atr_15m[i]) or np.isnan(zscore_15m[i]) or np.isnan(adx_15m[i]):
            signals[i] = 0.0
            continue
        
        trend = trend_15m[i]
        hist_val = hist_15m[i]
        hist_prev = hist_15m[i - 1] if i > 0 else 0
        zscore_val = zscore_15m[i]
        atr = atr_15m[i]
        adx_val = adx_15m[i]
        volume_confirmed = vol_spike_15m[i]
        price = close[i]
        
        # ATR filter - avoid trading when ATR is extremely high
        if atr > 0 and atr / price > 0.05:  # ATR > 5% of price = too volatile
            signals[i] = 0.0
            position_side[i] = 0
            entry_price[i] = 0
            tp_triggered[i] = 0
            continue
        
        # Check trailing stop and take profit for existing positions
        if i > 0 and position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else price
            prev_tp = tp_triggered[i - 1]
            prev_highest = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
            prev_lowest = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
            
            # Update highest/lowest since entry
            if prev_side == 1:
                current_highest = max(prev_highest, price)
                current_lowest = prev_lowest
            else:
                current_highest = prev_highest
                current_lowest = min(prev_lowest, price)
            
            highest_since_entry[i] = current_highest
            lowest_since_entry[i] = current_lowest
            
            if prev_side == 1:
                # Trailing stoploss (2*ATR from highest since entry)
                stoploss_price = current_highest - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R from entry)
                tp_price = prev_entry + TP_MULT * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    highest_since_entry[i] = current_highest
                    lowest_since_entry[i] = current_lowest
                    continue
                
                # MACD histogram exit signal (momentum fading)
                if hist_val < 0 and hist_prev >= 0:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
            elif prev_side == -1:
                # Trailing stoploss (2*ATR from lowest since entry)
                stoploss_price = current_lowest + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R from entry)
                tp_price = prev_entry - TP_MULT * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    highest_since_entry[i] = current_highest
                    lowest_since_entry[i] = current_lowest
                    continue
                
                # MACD histogram exit signal (momentum fading)
                if hist_val > 0 and hist_prev <= 0:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
            
            # Hold existing position
            signals[i] = signals[i - 1]
            position_side[i] = prev_side
            entry_price[i] = prev_entry
            tp_triggered[i] = prev_tp
            continue
        
        # Z-score filter - avoid entering at extreme deviations
        if abs(zscore_val) > ZSCORE_MAX:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # ADX filter - only trade strong trends
        if adx_val < ADX_MIN:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Dynamic position sizing based on ATR volatility
        current_atr_pct = atr / price if price > 0 else 0
        if current_atr_pct > 0:
            size_multiplier = min(1.5, max(0.5, TARGET_ATR_PCT / current_atr_pct))
        else:
            size_multiplier = 1.0
        
        position_size = SIZE_FULL * size_multiplier
        position_size = min(SIZE_FULL, max(SIZE_MIN, position_size))  # Clamp
        
        if trend == 1:  # 4h uptrend
            # MACD histogram cross above zero + volume confirmation
            if hist_val > MACD_LONG_THRESHOLD and hist_prev <= MACD_LONG_THRESHOLD:
                if volume_confirmed:
                    signals[i] = position_size
                    position_side[i] = 1
                    entry_price[i] = price
                    tp_triggered[i] = 0
                    highest_since_entry[i] = price
                    lowest_since_entry[i] = price
                else:
                    # Wait for volume confirmation
                    signals[i] = 0.0
                    position_side[i] = 0
            else:
                signals[i] = 0.0
                position_side[i] = 0
                    
        elif trend == -1:  # 4h downtrend
            # MACD histogram cross below zero + volume confirmation
            if hist_val < MACD_SHORT_THRESHOLD and hist_prev >= MACD_SHORT_THRESHOLD:
                if volume_confirmed:
                    signals[i] = -position_size
                    position_side[i] = -1
                    entry_price[i] = price
                    tp_triggered[i] = 0
                    highest_since_entry[i] = price
                    lowest_since_entry[i] = price
                else:
                    # Wait for volume confirmation
                    signals[i] = 0.0
                    position_side[i] = 0
            else:
                signals[i] = 0.0
                position_side[i] = 0
        else:  # No clear trend
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals