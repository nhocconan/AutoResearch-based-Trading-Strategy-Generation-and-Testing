#!/usr/bin/env python3
"""
EXPERIMENT #011 - HMA Trend + RSI Pullback + Z-Score Filter + ATR Stop
=======================================================================
Hypothesis: Hull Moving Average provides faster trend detection with less lag than
Donchian channels, while Z-score filter protects against extreme mean reversion.
Combined with RSI pullback entries and ATR-based trailing stops, this should capture
trends earlier while avoiding overextended entries.

Key differences from mtf_donchian_rsi_atr_v1:
- HMA(48) trend instead of Donchian (smoother, less whipsaw at turning points)
- Z-score(20) filter to avoid entering at extreme deviations (>2.5 std)
- Multi-timeframe: 4h HMA trend + 1h RSI entries + 1h Z-score filter
- ATR trailing stop with 2.5*ATR distance (same proven risk management)

Why this might beat Sharpe=5.884:
- HMA reacts faster to trend changes than Donchian breakout
- Z-score filter prevents buying tops/selling bottoms
- Same proven ATR stop management from #010
"""

import numpy as np
import pandas as pd

name = "mtf_hma_rsi_zscore_v1"
timeframe = "1h"
leverage = 1.0


def calculate_hma(close, period=48):
    """
    Calculate Hull Moving Average
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    Provides smoother trend with less lag than EMA
    """
    n = len(close)
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # Calculate WMA for half period
    wma_half = np.zeros(n)
    for i in range(half_period - 1, n):
        weights = np.arange(1, half_period + 1)
        wma_half[i] = np.sum(close[i - half_period + 1:i + 1] * weights) / np.sum(weights)
    
    # Calculate WMA for full period
    wma_full = np.zeros(n)
    for i in range(period - 1, n):
        weights = np.arange(1, period + 1)
        wma_full[i] = np.sum(close[i - period + 1:i + 1] * weights) / np.sum(weights)
    
    # Calculate raw HMA
    raw_hma = 2 * wma_half - wma_full
    
    # Calculate final HMA with sqrt period
    hma = np.zeros(n)
    for i in range(sqrt_period - 1, n):
        weights = np.arange(1, sqrt_period + 1)
        hma[i] = np.sum(raw_hma[i - sqrt_period + 1:i + 1] * weights) / np.sum(weights)
    
    return hma


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
    zscore = np.zeros(n)
    
    mean = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    mask = std > 0
    zscore[mask] = (close[mask] - mean[mask]) / std[mask]
    
    return zscore


def calculate_bb_width(close, period=20, std_mult=2.0):
    """Calculate Bollinger Band Width for volatility regime"""
    n = len(close)
    mean = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = mean + std_mult * std
    lower = mean - std_mult * std
    bb_width = (upper - lower) / mean
    
    return bb_width


def calculate_bb_percentile(close, period=20, std_mult=2.0, lookback=100):
    """Calculate BB Width percentile for regime detection"""
    bb_width = calculate_bb_width(close, period, std_mult)
    n = len(close)
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
    zscore_1h = calculate_zscore(close, period=20)
    bb_pct_1h = calculate_bb_percentile(close, period=20, std_mult=2.0, lookback=100)
    
    # 4h HMA for trend filter (resample 1h → 4h)
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
    
    # Calculate 4h HMA
    hma_4h = calculate_hma(c_4h, period=48)
    
    # 4h trend direction based on HMA slope and price position
    trend_4h = np.zeros(len(c_4h))
    for i in range(50, len(c_4h)):
        if hma_4h[i] > 0 and hma_4h[i-1] > 0:
            hma_slope = hma_4h[i] - hma_4h[i-1]
            price_vs_hma = (c_4h[i] - hma_4h[i]) / hma_4h[i]
            
            if hma_slope > 0 and price_vs_hma > -0.02:
                trend_4h[i] = 1  # Bullish (HMA rising, price near/above HMA)
            elif hma_slope < 0 and price_vs_hma < 0.02:
                trend_4h[i] = -1  # Bearish (HMA falling, price near/below HMA)
    
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
    
    # RSI thresholds for pullback entries
    RSI_LONG_ENTRY = 45   # Enter long on pullback in uptrend
    RSI_SHORT_ENTRY = 55  # Enter short on rally in downtrend
    
    # Z-score thresholds for extreme deviation filter
    ZSCORE_MAX = 2.5      # Don't enter if price > 2.5 std from mean
    ZSCORE_MIN = -2.5     # Don't enter if price < -2.5 std from mean
    
    # BB Width percentile thresholds for volatility filter
    BB_PCT_MIN = 0.20     # Don't trade in extremely low vol (consolidation)
    BB_PCT_MAX = 0.85     # Don't trade in extremely high vol (panic/euphoria)
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5
    
    first_valid = max(80, 50, 14, 20, 100)  # Wait for all indicators
    
    # Track entry prices for trailing stop logic
    entry_price_long = np.zeros(n)
    entry_price_short = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(rsi_1h[i]) or np.isnan(atr_1h[i]) or np.isnan(zscore_1h[i]) or np.isnan(bb_pct_1h[i]) or np.isnan(trend_1h[i]):
            signals[i] = 0.0
            continue
        
        trend = trend_1h[i]
        rsi_val = rsi_1h[i]
        zscore_val = zscore_1h[i]
        bb_pct = bb_pct_1h[i]
        atr = atr_1h[i]
        price = close[i]
        
        # Volatility filter - avoid extreme regimes
        if bb_pct < BB_PCT_MIN or bb_pct > BB_PCT_MAX:
            signals[i] = 0.0
            continue
        
        # ATR filter - avoid trading when ATR is extremely high
        if atr > 0 and atr / price > 0.05:  # ATR > 5% of price = too volatile
            signals[i] = 0.0
            continue
        
        # Z-score filter - avoid extreme deviations
        if abs(zscore_val) > ZSCORE_MAX:
            signals[i] = 0.0
            continue
        
        if trend == 1:  # 4h uptrend
            if rsi_val < RSI_LONG_ENTRY and zscore_val < ZSCORE_MAX:
                # Pullback entry - full position
                signals[i] = SIZE_FULL
                entry_price_long[i] = price
                highest_since_entry[i] = price
            elif rsi_val < 50 and zscore_val < ZSCORE_MAX:
                # Moderate pullback - half position
                signals[i] = SIZE_HALF
                entry_price_long[i] = price
                highest_since_entry[i] = price
            elif i > 0 and signals[i - 1] > 0:
                # Hold or trail existing long
                # Track highest price since entry
                entry_idx = max(0, i - 100)
                entry_prices = entry_price_long[entry_idx:i+1]
                valid_entries = entry_prices[entry_prices > 0]
                
                if len(valid_entries) > 0:
                    entry_price = valid_entries[0]
                    
                    # Update highest since entry
                    prices_since = close[entry_idx:i+1]
                    highest_since_entry[i] = max(np.max(prices_since), highest_since_entry[i-1] if i > 0 else price)
                    
                    # Trail stop: highest - 2.5*ATR
                    stoploss_price = highest_since_entry[i] - ATR_STOP_MULT * atr
                    
                    if price < stoploss_price:
                        signals[i] = 0.0  # Stoploss triggered
                    else:
                        signals[i] = signals[i - 1]  # Hold position
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
                
        elif trend == -1:  # 4h downtrend
            if rsi_val > RSI_SHORT_ENTRY and zscore_val > ZSCORE_MIN:
                # Rally entry - full short
                signals[i] = -SIZE_FULL
                entry_price_short[i] = price
                lowest_since_entry[i] = price
            elif rsi_val > 50 and zscore_val > ZSCORE_MIN:
                # Moderate rally - half short
                signals[i] = -SIZE_HALF
                entry_price_short[i] = price
                lowest_since_entry[i] = price
            elif i > 0 and signals[i - 1] < 0:
                # Hold or trail existing short
                # Track lowest price since entry
                entry_idx = max(0, i - 100)
                entry_prices = entry_price_short[entry_idx:i+1]
                valid_entries = entry_prices[entry_prices > 0]
                
                if len(valid_entries) > 0:
                    entry_price = valid_entries[0]
                    
                    # Update lowest since entry
                    prices_since = close[entry_idx:i+1]
                    lowest_since_entry[i] = min(np.min(prices_since), lowest_since_entry[i-1] if i > 0 else price)
                    
                    # Trail stop: lowest + 2.5*ATR
                    stoploss_price = lowest_since_entry[i] + ATR_STOP_MULT * atr
                    
                    if price > stoploss_price:
                        signals[i] = 0.0  # Stoploss triggered
                    else:
                        signals[i] = signals[i - 1]  # Hold position
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        else:  # No clear trend
            signals[i] = 0.0
    
    return signals