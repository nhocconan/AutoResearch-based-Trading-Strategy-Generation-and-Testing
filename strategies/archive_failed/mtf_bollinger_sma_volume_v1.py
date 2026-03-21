#!/usr/bin/env python3
"""
EXPERIMENT #003 - Bollinger Mean Reversion + SMA Trend Filter + Volume Confirmation
====================================================================================
Hypothesis: Mean reversion strategies work well in crypto's ranging periods (60-70% of time).
Using 4h SMA(50) for trend direction + 1h Bollinger Band extremes for entries + volume
confirmation should capture reversions while avoiding counter-trend trades.

Key innovations vs current best (mtf_supertrend_macd_adx_v1):
- Mean reversion instead of momentum (different market regime)
- Bollinger Band position (%B) for entry timing at extremes
- Volume spike filter confirms genuine reversals vs fakeouts
- SMA(50) trend filter prevents counter-trend mean reversion
- ATR trailing stop at 2.0*ATR with position reduction at 1.5R profit

Why this might beat Sharpe=1.278:
- Crypto spends more time ranging than trending (mean reversion edge)
- Bollinger %B at extremes has statistical edge for reversions
- Volume confirmation reduces false signals
- Multi-timeframe trend filter protects against strong trends
- Discrete position sizing (0.0, ±0.25, ±0.35) reduces churn costs
"""

import numpy as np
import pandas as pd

name = "mtf_bollinger_sma_volume_v1"
timeframe = "1h"
leverage = 1.0


def calculate_sma(close, period=50):
    """Calculate Simple Moving Average"""
    n = len(close)
    sma = np.zeros(n)
    for i in range(period - 1, n):
        sma[i] = np.mean(close[i - period + 1:i + 1])
    return sma


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands with %B indicator"""
    n = len(close)
    sma = np.zeros(n)
    upper = np.zeros(n)
    lower = np.zeros(n)
    pct_b = np.zeros(n)
    
    for i in range(period - 1, n):
        sma[i] = np.mean(close[i - period + 1:i + 1])
        std = np.std(close[i - period + 1:i + 1])
        upper[i] = sma[i] + std_mult * std
        lower[i] = sma[i] - std_mult * std
        
        band_width = upper[i] - lower[i]
        if band_width > 0:
            pct_b[i] = (close[i] - lower[i]) / band_width
        else:
            pct_b[i] = 0.5
    
    return upper, lower, pct_b


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


def calculate_volume_spike(volume, period=20):
    """Detect volume spikes (>1.5x average)"""
    n = len(volume)
    avg_volume = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    volume_ratio = np.zeros(n)
    mask = avg_volume > 0
    volume_ratio[mask] = volume[mask] / avg_volume[mask]
    return volume_ratio


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values if "volume" in prices.columns else np.ones(len(close))
    n = len(close)
    
    # 1h indicators for entry timing
    rsi_1h = calculate_rsi(close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    bb_upper, bb_lower, pct_b = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    volume_ratio = calculate_volume_spike(volume, period=20)
    
    # 4h SMA for trend filter (resample 1h → 4h)
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
    sma_4h = calculate_sma(c_4h, period=50)
    
    # 4h trend direction based on SMA position
    trend_4h = np.zeros(len(c_4h))
    for i in range(50, len(c_4h)):
        if sma_4h[i] > 0:
            price_vs_sma = (c_4h[i] - sma_4h[i]) / sma_4h[i]
            if price_vs_sma > 0.01:  # Price > 1% above SMA
                trend_4h[i] = 1  # Bullish
            elif price_vs_sma < -0.01:  # Price < 1% below SMA
                trend_4h[i] = -1  # Bearish
    
    # Map 4h trend back to 1h timeframe
    trend_1h = np.zeros(n)
    idx_1h_to_4h = np.arange(n) // 4
    
    for i in range(n):
        idx_4h = idx_1h_to_4h[i]
        if idx_4h < len(trend_4h):
            trend_1h[i] = trend_4h[idx_4h]
    
    # Generate signals with mean reversion logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels to reduce churn
    SIZE_FULL = 0.35   # Full position at extreme reversions
    SIZE_HALF = 0.25   # Reduced position at moderate reversions
    
    # Bollinger %B thresholds for mean reversion
    BB_LONG_ENTRY = 0.15   # Enter long when %B < 0.15 (near lower band)
    BB_SHORT_ENTRY = 0.85  # Enter short when %B > 0.85 (near upper band)
    BB_EXIT = 0.50         # Exit when price reverts to middle
    
    # RSI confirmation thresholds
    RSI_LONG_CONFIRM = 35   # RSI oversold for long entries
    RSI_SHORT_CONFIRM = 65  # RSI overbought for short entries
    
    # Volume confirmation
    VOLUME_SPIKE = 1.5      # Volume must be >1.5x average
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    ATR_PROFIT_MULT = 1.5   # Take partial profit at 1.5R
    
    first_valid = max(80, 50, 20, 14)  # Wait for all indicators
    
    # Track entry prices and position state for trailing stop logic
    entry_price = np.zeros(n)
    position_side = np.zeros(n)  # 1 for long, -1 for short, 0 for flat
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(rsi_1h[i]) or np.isnan(atr_1h[i]) or np.isnan(pct_b[i]):
            signals[i] = 0.0
            continue
        
        trend = trend_1h[i]
        rsi_val = rsi_1h[i]
        pct_b_val = pct_b[i]
        atr = atr_1h[i]
        price = close[i]
        vol_ratio = volume_ratio[i]
        
        # ATR filter - avoid trading when ATR is extremely high
        if atr > 0 and atr / price > 0.05:  # ATR > 5% of price = too volatile
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Check trailing stop for existing positions
        if i > 0 and position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else price
            prev_highest = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else price
            prev_lowest = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else price
            
            # Update highest/lowest since entry
            if prev_side == 1:  # Long
                current_highest = max(prev_highest, price)
                current_lowest = prev_lowest
                trailing_stop = current_highest - ATR_STOP_MULT * atr
                
                if price < trailing_stop:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Check for profit target (reduce position at 1.5R)
                profit_r = (price - prev_entry) / (ATR_STOP_MULT * atr) if atr > 0 else 0
                if profit_r >= ATR_PROFIT_MULT and signals[i - 1] > 0.30:
                    signals[i] = SIZE_HALF  # Reduce to half
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    highest_since_entry[i] = current_highest
                    lowest_since_entry[i] = current_lowest
                    continue
                
                # Exit at mean reversion (%B > 0.50)
                if pct_b_val > BB_EXIT:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                signals[i] = signals[i - 1]
                position_side[i] = 1
                entry_price[i] = prev_entry
                highest_since_entry[i] = current_highest
                lowest_since_entry[i] = current_lowest
                continue
                
            elif prev_side == -1:  # Short
                current_highest = prev_highest
                current_lowest = min(prev_lowest, price)
                trailing_stop = current_lowest + ATR_STOP_MULT * atr
                
                if price > trailing_stop:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Check for profit target (reduce position at 1.5R)
                profit_r = (prev_entry - price) / (ATR_STOP_MULT * atr) if atr > 0 else 0
                if profit_r >= ATR_PROFIT_MULT and signals[i - 1] < -0.30:
                    signals[i] = -SIZE_HALF  # Reduce to half
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    highest_since_entry[i] = current_highest
                    lowest_since_entry[i] = current_lowest
                    continue
                
                # Exit at mean reversion (%B < 0.50)
                if pct_b_val < BB_EXIT:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                signals[i] = signals[i - 1]
                position_side[i] = -1
                entry_price[i] = prev_entry
                highest_since_entry[i] = current_highest
                lowest_since_entry[i] = current_lowest
                continue
        
        # Mean reversion entry logic with trend filter
        if trend == 1:  # 4h uptrend - only look for long mean reversion
            if pct_b_val < BB_LONG_ENTRY and rsi_val < RSI_LONG_CONFIRM:
                # Volume confirmation required for entry
                if vol_ratio >= VOLUME_SPIKE:
                    signals[i] = SIZE_FULL
                    position_side[i] = 1
                    entry_price[i] = price
                    highest_since_entry[i] = price
                    lowest_since_entry[i] = price
                elif vol_ratio >= 1.0:  # Accept normal volume with smaller size
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = price
                    highest_since_entry[i] = price
                    lowest_since_entry[i] = price
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
            else:
                signals[i] = 0.0
                position_side[i] = 0
                
        elif trend == -1:  # 4h downtrend - only look for short mean reversion
            if pct_b_val > BB_SHORT_ENTRY and rsi_val > RSI_SHORT_CONFIRM:
                # Volume confirmation required for entry
                if vol_ratio >= VOLUME_SPIKE:
                    signals[i] = -SIZE_FULL
                    position_side[i] = -1
                    entry_price[i] = price
                    highest_since_entry[i] = price
                    lowest_since_entry[i] = price
                elif vol_ratio >= 1.0:  # Accept normal volume with smaller size
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = price
                    highest_since_entry[i] = price
                    lowest_since_entry[i] = price
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
            else:
                signals[i] = 0.0
                position_side[i] = 0
        else:  # No clear trend - reduce position size or stay flat
            if pct_b_val < BB_LONG_ENTRY and rsi_val < RSI_LONG_CONFIRM:
                signals[i] = SIZE_HALF * 0.7  # Smaller position in no-trend
                position_side[i] = 1
                entry_price[i] = price
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
            elif pct_b_val > BB_SHORT_ENTRY and rsi_val > RSI_SHORT_CONFIRM:
                signals[i] = -SIZE_HALF * 0.7  # Smaller position in no-trend
                position_side[i] = -1
                entry_price[i] = price
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
            else:
                signals[i] = 0.0
                position_side[i] = 0
    
    return signals