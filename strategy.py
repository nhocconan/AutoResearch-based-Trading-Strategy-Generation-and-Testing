#!/usr/bin/env python3
"""
EXPERIMENT #009 - KAMA Trend + BB Squeeze Breakout + Volume Confirmation
=======================================================================
Hypothesis: Bollinger Band squeeze breakouts capture volatility expansion early,
combined with KAMA adaptive trend filter and volume confirmation to reduce false breakouts.
This differs from current best by using volatility regime (BB squeeze) instead of RSI pullback.

Key innovations:
- 4h KAMA for adaptive trend (responds to volatility changes better than HMA)
- 1h BB squeeze detection (low vol → expansion = breakout opportunity)
- Volume spike confirmation (1.5x average volume validates breakout)
- ATR trailing stop for dynamic risk management
- Z-score filter to avoid extreme overbought/oversold entries

Why this might beat Sharpe=2.931:
- BB squeeze captures momentum before price moves significantly
- Volume confirmation reduces false breakout whipsaw
- KAMA adapts to regime changes faster than fixed MA
"""

import numpy as np
import pandas as pd

name = "mtf_kama_bbsqueeze_volume_v2"
timeframe = "1h"
leverage = 1.0


def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts smoothing based on market noise/volatility
    """
    n = len(close)
    kama = np.zeros(n)
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(er_period, n):
        signal = abs(close[i] - close[i - er_period])
        noise = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if noise > 0:
            er[i] = signal / noise
    
    # Calculate smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
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


def calculate_bb_squeeze(close, period=20, std_mult=2.0, bb_period=20, kc_mult=1.5):
    """
    Detect Bollinger Band squeeze (BB inside Keltner Channel)
    Returns: squeeze_active (bool), bb_width, position within bands
    """
    n = len(close)
    
    # Bollinger Bands
    bb_mean = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    bb_upper = bb_mean + std_mult * bb_std
    bb_lower = bb_mean - std_mult * bb_std
    bb_width = (bb_upper - bb_lower) / bb_mean
    
    # Keltner Channels (using ATR)
    atr = calculate_atr(
        pd.Series(close).values,
        pd.Series(close).values,
        close,
        period=bb_period
    )
    # Approximate high/low for ATR calculation
    kc_upper = bb_mean + kc_mult * atr
    kc_lower = bb_mean - kc_mult * atr
    
    # Squeeze detection: BB inside KC
    squeeze_active = np.zeros(n, dtype=bool)
    for i in range(bb_period, n):
        if bb_upper[i] < kc_upper[i] and bb_lower[i] > kc_lower[i]:
            squeeze_active[i] = True
    
    # Position within BB (0=lower, 0.5=middle, 1=upper)
    bb_position = np.zeros(n)
    for i in range(bb_period, n):
        if bb_upper[i] > bb_lower[i]:
            bb_position[i] = (close[i] - bb_lower[i]) / (bb_upper[i] - bb_lower[i])
    
    return squeeze_active, bb_width, bb_position, bb_upper, bb_lower


def calculate_volume_spike(volume, period=20, threshold=1.5):
    """Detect volume spikes above moving average"""
    n = len(volume)
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_spike = volume > (vol_ma * threshold)
    return vol_spike, vol_ma


def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion filter"""
    n = len(close)
    mean = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    zscore = np.zeros(n)
    mask = std > 0
    zscore[mask] = (close[mask] - mean[mask]) / std[mask]
    
    return zscore


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


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices.get("volume", np.ones(len(close))).values
    n = len(close)
    
    # 1h indicators for entry timing
    rsi_1h = calculate_rsi(close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    squeeze_1h, bb_width_1h, bb_pos_1h, bb_upper_1h, bb_lower_1h = calculate_bb_squeeze(close)
    vol_spike_1h, vol_ma_1h = calculate_volume_spike(volume, period=20, threshold=1.5)
    zscore_1h = calculate_zscore(close, period=20)
    
    # 4h KAMA for trend filter (resample 1h → 4h)
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
    h_4h = df_4h['high'].values
    l_4h = df_4h['low'].values
    
    # Calculate 4h KAMA
    kama_4h = calculate_kama(c_4h, er_period=10, fast_period=2, slow_period=30)
    
    # 4h trend direction based on KAMA slope and price position
    trend_4h = np.zeros(len(c_4h))
    for i in range(40, len(c_4h)):
        if kama_4h[i] > 0 and kama_4h[i-1] > 0:
            kama_slope = (kama_4h[i] - kama_4h[i-1]) / kama_4h[i-1]
            price_vs_kama = (c_4h[i] - kama_4h[i]) / kama_4h[i]
            
            if kama_slope > 0.001 and price_vs_kama > -0.02:
                trend_4h[i] = 1  # Bullish
            elif kama_slope < -0.001 and price_vs_kama < 0.02:
                trend_4h[i] = -1  # Bearish
    
    # Map 4h trend back to 1h timeframe
    trend_1h = np.zeros(n)
    idx_1h_to_4h = np.arange(n) // 4
    
    for i in range(n):
        idx_4h = min(idx_1h_to_4h[i], len(trend_4h) - 1)
        if idx_4h >= 0:
            trend_1h[i] = trend_4h[idx_4h]
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels to reduce churn
    SIZE_FULL = 0.35   # Full position with all confirmations
    SIZE_HALF = 0.20   # Reduced position with partial confirmations
    
    # BB squeeze breakout thresholds
    BB_WIDTH_LOW = 0.02   # Squeeze threshold (narrow bands)
    BB_WIDTH_HIGH = 0.08  # Expansion threshold
    
    # Z-score filter thresholds
    ZSCORE_MAX = 2.0      # Don't enter if extremely overbought/oversold
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5
    
    first_valid = max(80, 40, 20, 14)  # Wait for all indicators
    
    # Track entry prices for trailing stop
    long_entry_price = np.zeros(n)
    short_entry_price = np.zeros(n)
    long_highest = np.zeros(n)
    short_lowest = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(rsi_1h[i]) or np.isnan(atr_1h[i]) or np.isnan(bb_width_1h[i]):
            signals[i] = 0.0
            continue
        
        trend = trend_1h[i]
        rsi_val = rsi_1h[i]
        bb_width = bb_width_1h[i]
        bb_pos = bb_pos_1h[i]
        atr = atr_1h[i]
        price = close[i]
        zscore = zscore_1h[i]
        vol_confirmed = vol_spike_1h[i]
        
        # Z-score filter - avoid extreme entries
        if abs(zscore) > ZSCORE_MAX:
            signals[i] = 0.0
            continue
        
        # ATR filter - avoid trading when ATR is extremely high
        if atr > 0 and atr / price > 0.05:  # ATR > 5% of price = too volatile
            signals[i] = 0.0
            continue
        
        # Check existing positions for trailing stop
        if signals[i - 1] > 0:  # Existing long
            # Update highest price since entry
            if i > 0:
                long_highest[i] = max(long_highest[i-1], price)
            else:
                long_highest[i] = price
            
            # Find entry price
            if long_entry_price[i-1] > 0:
                entry_price = long_entry_price[i-1]
            else:
                entry_price = price
            
            # Trailing stop: exit if price drops 2.5*ATR from highest
            stoploss_price = long_highest[i] - ATR_STOP_MULT * atr
            
            if price < stoploss_price:
                signals[i] = 0.0  # Stoploss triggered
                long_entry_price[i] = 0.0
                long_highest[i] = 0.0
            else:
                # Hold position, maybe reduce at profit target
                profit_pct = (price - entry_price) / entry_price if entry_price > 0 else 0
                if profit_pct > 0.04:  # 4% profit = reduce to half
                    signals[i] = SIZE_HALF
                else:
                    signals[i] = signals[i - 1]
                long_entry_price[i] = entry_price
            continue
        
        elif signals[i - 1] < 0:  # Existing short
            # Update lowest price since entry
            if i > 0:
                short_lowest[i] = min(short_lowest[i-1], price)
            else:
                short_lowest[i] = price
            
            # Find entry price
            if short_entry_price[i-1] > 0:
                entry_price = short_entry_price[i-1]
            else:
                entry_price = price
            
            # Trailing stop: exit if price rises 2.5*ATR from lowest
            stoploss_price = short_lowest[i] + ATR_STOP_MULT * atr
            
            if price > stoploss_price:
                signals[i] = 0.0  # Stoploss triggered
                short_entry_price[i] = 0.0
                short_lowest[i] = 0.0
            else:
                # Hold position, maybe reduce at profit target
                profit_pct = (entry_price - price) / entry_price if entry_price > 0 else 0
                if profit_pct > 0.04:  # 4% profit = reduce to half
                    signals[i] = -SIZE_HALF
                else:
                    signals[i] = signals[i - 1]
                short_entry_price[i] = entry_price
            continue
        
        # No existing position - look for new entries
        if trend == 1:  # 4h uptrend - look for long entries
            # BB squeeze breakout: narrow bands + price breaking upper + volume confirmation
            squeeze_active = squeeze_1h[i] or (i > 0 and squeeze_1h[i-1])
            breakout_long = bb_pos > 0.75  # Price in upper 25% of BB
            
            if squeeze_active and breakout_long:
                if vol_confirmed:
                    signals[i] = SIZE_FULL
                    long_entry_price[i] = price
                    long_highest[i] = price
                elif bb_width < BB_WIDTH_LOW:
                    # Squeeze but no volume - half position
                    signals[i] = SIZE_HALF
                    long_entry_price[i] = price
                    long_highest[i] = price
                else:
                    signals[i] = 0.0
            elif bb_width < BB_WIDTH_LOW and bb_pos > 0.5:
                # Early squeeze position (before breakout)
                signals[i] = SIZE_HALF
                long_entry_price[i] = price
                long_highest[i] = price
            else:
                signals[i] = 0.0
                
        elif trend == -1:  # 4h downtrend - look for short entries
            # BB squeeze breakout: narrow bands + price breaking lower + volume confirmation
            squeeze_active = squeeze_1h[i] or (i > 0 and squeeze_1h[i-1])
            breakout_short = bb_pos < 0.25  # Price in lower 25% of BB
            
            if squeeze_active and breakout_short:
                if vol_confirmed:
                    signals[i] = -SIZE_FULL
                    short_entry_price[i] = price
                    short_lowest[i] = price
                elif bb_width < BB_WIDTH_LOW:
                    # Squeeze but no volume - half position
                    signals[i] = -SIZE_HALF
                    short_entry_price[i] = price
                    short_lowest[i] = price
                else:
                    signals[i] = 0.0
            elif bb_width < BB_WIDTH_LOW and bb_pos < 0.5:
                # Early squeeze position (before breakout)
                signals[i] = -SIZE_HALF
                short_entry_price[i] = price
                short_lowest[i] = price
            else:
                signals[i] = 0.0
        else:  # No clear trend
            signals[i] = 0.0
    
    return signals