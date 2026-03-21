#!/usr/bin/env python3
"""
Experiment #266: 30m RSI Mean Reversion with 4h/1d HMA Trend Filter
Hypothesis: 30m timeframe captures intraday swings better than 12h/1d. RSI mean reversion
works well in range/bear markets (2025 test period). 4h HMA provides trend bias without
being too restrictive. 1d HMA confirms macro direction. Entry on RSI extremes (30/70) with
trend alignment generates sufficient trades. Simple 2*ATR stoploss. Position sizing: 0.25
entry, 0.125 half at 2R profit. This differs from failed experiments by using simpler
entry conditions that actually trigger (RSI 30/70 vs strict ADX/CHOP filters).
Target: Beat Sharpe=0.499 with more consistent trades across all symbols.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_rsi_mr_4h_1d_hma_trend_atr_v1"
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
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate RSI indicator."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean()
    std = pd.Series(close).rolling(window=period, min_periods=period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper.values, lower.values, sma.values

def calculate_volume_ratio(taker_buy_volume, volume):
    """Calculate taker buy volume ratio (0-1, >0.5 = bullish)."""
    ratio = np.where(volume > 0, taker_buy_volume / volume, 0.5)
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_volume = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    sma_50 = calculate_sma(close, 50)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    vol_ratio = calculate_volume_ratio(taker_buy_volume, volume)
    
    # Track previous values
    prev_rsi = np.roll(rsi, 1)
    prev_rsi[0] = rsi[0]
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.125
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # HTF trend filters (simple and loose to ensure trades)
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # RSI mean reversion signals (loose thresholds for trade generation)
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_rising = rsi[i] > prev_rsi[i]
        rsi_falling = rsi[i] < prev_rsi[i]
        
        # Bollinger Band position
        near_lower = close[i] < bb_lower[i] * 1.01
        near_upper = close[i] > bb_upper[i] * 0.99
        below_mid = close[i] < bb_mid[i]
        above_mid = close[i] > bb_mid[i]
        
        # SMA 50 position
        above_sma50 = close[i] > sma_50[i]
        below_sma50 = close[i] < sma_50[i]
        
        # Volume confirmation
        vol_bullish = vol_ratio[i] > 0.50
        vol_bearish = vol_ratio[i] < 0.50
        
        # Price momentum
        price_rising = close[i] > prev_close[i]
        price_falling = close[i] < prev_close[i]
        
        new_signal = 0.0
        
        # === LONG ENTRY (Mean Reversion in Uptrend) ===
        # Primary: RSI oversold + 4h bullish trend
        if rsi_oversold and trend_4h_bullish:
            if rsi_rising or price_rising:
                new_signal = SIZE_ENTRY
        
        # Secondary: RSI oversold + near BB lower + volume support
        elif rsi_oversold and near_lower:
            if vol_bullish or above_sma50:
                new_signal = SIZE_ENTRY
        
        # Tertiary: Pullback to SMA50 in uptrend
        elif trend_4h_bullish and trend_1d_bullish:
            if below_sma50 and prev_close[i] < sma_50[i] and close[i] > sma_50[i]:
                if rsi[i] < 50:
                    new_signal = SIZE_ENTRY
        
        # === SHORT ENTRY (Mean Reversion in Downtrend) ===
        # Primary: RSI overbought + 4h bearish trend
        if rsi_overbought and trend_4h_bearish:
            if rsi_falling or price_falling:
                new_signal = -SIZE_ENTRY
        
        # Secondary: RSI overbought + near BB upper + volume support
        elif rsi_overbought and near_upper:
            if vol_bearish or below_sma50:
                new_signal = -SIZE_ENTRY
        
        # Tertiary: Rally to SMA50 in downtrend
        elif trend_4h_bearish and trend_1d_bearish:
            if above_sma50 and prev_close[i] > sma_50[i] and close[i] < sma_50[i]:
                if rsi[i] > 50:
                    new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2*ATR from highest)
            current_stop = highest_close - 2.0 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.0 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2*ATR from lowest)
            current_stop = lowest_close + 2.0 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.0 * atr[i]
                profit = entry_price - close[i]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reduced (take profit)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals