#!/usr/bin/env python3
"""
Experiment #428: 30m Bollinger-Keltner Squeeze + 4h HMA Trend + Volume Confirmation

Hypothesis: After 11 consecutive failures, the pattern is clear - pure trend 
following and pure mean reversion both fail on crypto. The key is capturing 
VOLATILITY EXPANSION events which occur in both bull and bear markets.

This strategy combines:
1. BOLLINGER-KELTNER SQUEEZE: When BB width < Keltner width, market is 
   coiling. Breakout from squeeze = high probability move.
2. 4h HMA TREND BIAS: Only take long breakouts when price > 4h HMA, only 
   short when price < 4h HMA. This filters counter-trend squeezes.
3. VOLUME CONFIRMATION: Breakout must have volume > 1.5x 20-bar average.
   Filters false breakouts which killed strategy #416.
4. ATR TRAILING STOP: 2.5x ATR from entry, protects from reversals.
5. CONSERVATIVE SIZING: 0.25 position size (25% capital max).

Why 30m:
- Fast enough to catch squeeze breakouts early
- Slow enough to avoid 5m/15m noise
- 4h HTF provides good trend context (8 bars per 4h)

Expected: 30-50 trades/year, Sharpe > 0.7, DD < -30%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_bb_keltner_squeeze_4h_hma_vol_atr_v1"
timeframe = "30m"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    return upper.values, lower.values, sma.values

def calculate_keltner_channel(high, low, close, period=20, atr_period=14, multiplier=1.5):
    """Calculate Keltner Channel."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    atr = calculate_atr(high, low, close, atr_period)
    upper = ema.values + (multiplier * atr)
    lower = ema.values - (multiplier * atr)
    return upper, lower

def calculate_volume_ma(volume, period=20):
    """Calculate volume moving average."""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean()
    return vol_ma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    keltner_upper, keltner_lower = calculate_keltner_channel(high, low, close, 20, 14, 1.5)
    vol_ma = calculate_volume_ma(volume, 20)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(vol_ma[i]) or vol_ma[i] == 0:
            signals[i] = 0.0
            continue
        
        # === SQUEEZE DETECTION ===
        # Squeeze = BB inside Keltner (low volatility, coiling)
        bb_width = bb_upper[i] - bb_lower[i]
        keltner_width = keltner_upper[i] - keltner_lower[i]
        
        in_squeeze = bb_width < keltner_width
        
        # Squeeze was active in previous bars (look back 5 bars)
        squeeze_history = False
        for j in range(max(100, i-5), i):
            bb_w = bb_upper[j] - bb_lower[j]
            k_w = keltner_upper[j] - keltner_lower[j]
            if bb_w < k_w:
                squeeze_history = True
                break
        
        # === 4h HMA TREND BIAS ===
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        volume_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        high_volume = volume_ratio > 1.5
        
        # === BREAKOUT SIGNALS ===
        # Long breakout: price breaks above BB upper + was in squeeze + volume + bull trend
        breakout_long = (close[i] > bb_upper[i-1] and 
                        squeeze_history and 
                        high_volume and 
                        bull_trend_4h)
        
        # Short breakout: price breaks below BB lower + was in squeeze + volume + bear trend
        breakout_short = (close[i] < bb_lower[i-1] and 
                         squeeze_history and 
                         high_volume and 
                         bear_trend_4h)
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        if breakout_long:
            new_signal = SIZE
        elif breakout_short:
            new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_4h:
                new_signal = 0.0
            if position_side < 0 and bull_trend_4h:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals