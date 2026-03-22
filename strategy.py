#!/usr/bin/env python3
"""
Experiment #011: 12h RSI Mean Reversion + 1d HMA Trend Filter
Hypothesis: 12h timeframe captures multi-day swings while 1d HMA provides trend bias.
RSI(14) with moderate thresholds (35/65) generates sufficient trades vs extreme (10/90).
Bollinger Band position adds context - enter long near lower band in uptrend, short near upper in downtrend.
ATR trailing stop at 2.5*ATR accommodates 12h volatility.
Timeframe: 12h (REQUIRED), HTF: 1d via mtf_data helper.
Position sizing: 0.30 base, discrete levels (0.0, ±0.30) to minimize fee churn.
Key innovation: Moderate RSI thresholds ensure 10+ trades per symbol while 1d trend filter avoids counter-trend traps.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_rsi_1d_hma_bb_v1"
timeframe = "12h"
leverage = 1.0

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    n = len(close)
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    if n < period + 1:
        return rsi
    
    # Calculate price changes
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    # Initial SMA for first RSI value
    avg_gain = np.mean(gains[1:period+1])
    avg_loss = np.mean(losses[1:period+1])
    
    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100 - (100 / (1 + rs))
    
    # Wilder's smoothing for remaining values
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    n = len(close)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = np.zeros(n)
    atr[:] = np.nan
    
    if n < period:
        return atr
    
    # Initial SMA
    atr[period-1] = np.mean(tr[:period])
    
    # Wilder's smoothing
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    
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

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and bandwidth."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = (upper - lower) / sma
    bandwidth[np.isnan(bandwidth)] = 0.0
    # Calculate position within bands (0=lower, 0.5=middle, 1=upper)
    bb_position = (close - lower) / (upper - lower + 1e-10)
    bb_position = np.clip(bb_position, 0, 1)
    bb_position[np.isnan(bb_position)] = 0.5
    return upper, lower, bandwidth, sma, bb_position

def calculate_volume_sma(volume, period=20):
    """Calculate volume SMA for volume confirmation."""
    vol_s = pd.Series(volume)
    vol_sma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, period=14)
    bb_upper, bb_lower, bb_bandwidth, bb_sma, bb_position = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    vol_sma = calculate_volume_sma(volume, period=20)
    
    # Additional trend filter on 12h
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_200 = pd.Series(close).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels to minimize fee churn (Rule 4)
    SIZE_BASE = 0.30
    SIZE_EXIT = 0.0
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_bandwidth[i]) or np.isnan(bb_sma[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(vol_sma[i]) or vol_sma[i] == 0:
            signals[i] = 0.0
            continue
        
        # 1d trend bias (HTF)
        bull_trend = close[i] > hma_1d_aligned[i]
        bear_trend = close[i] < hma_1d_aligned[i]
        
        # RSI signals - MODERATE thresholds for sufficient trades
        rsi_oversold = rsi[i] < 40  # Not too extreme
        rsi_overbought = rsi[i] > 60  # Not too extreme
        rsi_neutral = 40 <= rsi[i] <= 60
        
        # Price position vs Bollinger Bands
        price_near_lower = bb_position[i] < 0.15  # Bottom 15% of bands
        price_near_upper = bb_position[i] > 0.85  # Top 15% of bands
        price_near_middle = 0.35 < bb_position[i] < 0.65
        
        # Volume confirmation
        volume_above_avg = volume[i] > vol_sma[i] * 0.8  # At least 80% of avg
        
        # EMA trend confirmation on 12h
        ema_bullish = close[i] > ema_50[i]
        ema_bearish = close[i] < ema_50[i]
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Primary: RSI oversold + price near lower BB + 1d bull trend
        if rsi_oversold and price_near_lower and bull_trend and volume_above_avg:
            new_signal = SIZE_BASE
        # Secondary: RSI oversold + 1d bull trend + price above 12h EMA50
        elif rsi_oversold and bull_trend and ema_bullish and volume_above_avg:
            new_signal = SIZE_BASE
        # Tertiary: RSI rising from oversold + 1d bull trend
        elif rsi[i] < 45 and rsi[i] > rsi[i-1] and bull_trend:
            new_signal = SIZE_BASE
        
        # === SHORT ENTRY ===
        # Primary: RSI overbought + price near upper BB + 1d bear trend
        if rsi_overbought and price_near_upper and bear_trend and volume_above_avg:
            new_signal = -SIZE_BASE
        # Secondary: RSI overbought + 1d bear trend + price below 12h EMA50
        elif rsi_overbought and bear_trend and ema_bearish and volume_above_avg:
            new_signal = -SIZE_BASE
        # Tertiary: RSI falling from overbought + 1d bear trend
        elif rsi[i] > 55 and rsi[i] < rsi[i-1] and bear_trend:
            new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Long position stoploss
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for 12h timeframe)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        # Short position stoploss
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for 12h timeframe)
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals