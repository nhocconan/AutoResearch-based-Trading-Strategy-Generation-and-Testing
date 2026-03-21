#!/usr/bin/env python3
"""
Experiment #029: 12h KAMA Trend + Daily HMA Regime + RSI Momentum
Hypothesis: 12h timeframe captures multi-day swings with less noise than intraday.
KAMA (Kaufman Adaptive MA) adapts to volatility - fast in trends, slow in chop.
Daily HMA provides regime filter (bull/bear) without lag of SMA.
RSI momentum filter ensures entry with momentum, not against it.
Multiple entry triggers (KAMA cross, RSI extreme, trend continuation) ensure ≥10 trades.
Position sizing 0.25 with 2.5x ATR stoploss protects against crashes.
Relaxed RSI thresholds (35-65 range) to avoid 0-trade failure while filtering extremes.
Volume confirmation reduces false breakouts.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_daily_rsi_vol_v1"
timeframe = "12h"
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

def calculate_kama(close, period=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average."""
    close_s = pd.Series(close)
    # Efficiency Ratio
    change = np.abs(close - np.roll(close, period))
    change[0:period] = np.abs(close[0:period] - close[0])
    volatility = pd.Series(np.abs(close - np.roll(close, 1))).rolling(window=period, min_periods=period).sum().values
    volatility[0] = change[0]
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    # Smoothing constant
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    # KAMA calculation
    kama = np.zeros(len(close))
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
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

def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD indicator."""
    close_s = pd.Series(close)
    ema_fast = close_s.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = close_s.ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line.values, signal_line.values, histogram.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load daily HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    kama_10 = calculate_kama(close, 10, 2, 30)
    kama_30 = calculate_kama(close, 30, 2, 30)
    macd_line, macd_signal, macd_hist = calculate_macd(close, 12, 26, 9)
    
    # Volume SMA for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_sma = np.nan_to_num(vol_sma, nan=np.nanmean(volume))
    
    # Price relative to KAMA
    price_kama_ratio = close / kama_10
    
    signals = np.zeros(n)
    SIZE = 0.25
    HALF_SIZE = 0.125
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Daily trend filter (major regime)
        daily_bullish = hma_1d_aligned[i] > 0 and close[i] > hma_1d_aligned[i]
        daily_bearish = hma_1d_aligned[i] > 0 and close[i] < hma_1d_aligned[i]
        
        # KAMA trend direction
        kama_trend_long = kama_10[i] > kama_30[i]
        kama_trend_short = kama_10[i] < kama_30[i]
        
        # KAMA cross signals
        kama_cross_long = kama_10[i] > kama_30[i] and kama_10[i-1] <= kama_30[i-1]
        kama_cross_short = kama_10[i] < kama_30[i] and kama_10[i-1] >= kama_30[i-1]
        
        # MACD signals
        macd_bullish = macd_hist[i] > 0 and macd_hist[i-1] <= 0
        macd_bearish = macd_hist[i] < 0 and macd_hist[i-1] >= 0
        macd_positive = macd_hist[i] > 0
        macd_negative = macd_hist[i] < 0
        
        # RSI momentum (relaxed for more trades)
        rsi_bullish = rsi[i] > 40 and rsi[i] < 70
        rsi_bearish = rsi[i] > 30 and rsi[i] < 60
        rsi_rising = rsi[i] > rsi[i-3] if i > 3 else True
        rsi_falling = rsi[i] < rsi[i-3] if i > 3 else True
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        
        # Volume confirmation
        vol_confirm = volume[i] > vol_sma[i] * 0.9 if vol_sma[i] > 0 else True
        
        # Price position relative to KAMA
        price_above_kama = close[i] > kama_10[i]
        price_below_kama = close[i] < kama_10[i]
        
        # Entry logic - MULTIPLE triggers to ensure trades (Rule 9)
        new_signal = 0.0
        
        # LONG ENTRY TRIGGERS (any one can trigger)
        # Trigger 1: KAMA cross long with daily support
        if kama_cross_long and (daily_bullish or rsi_bullish):
            new_signal = SIZE
        # Trigger 2: KAMA trend + MACD bullish + RSI ok (trend continuation)
        elif kama_trend_long and macd_positive and rsi_bullish and price_above_kama:
            new_signal = SIZE
        # Trigger 3: Daily bullish + KAMA trend + volume (regime + trend)
        elif daily_bullish and kama_trend_long and vol_confirm:
            new_signal = SIZE
        # Trigger 4: RSI from oversold with KAMA support (mean reversion in trend)
        elif rsi_oversold and kama_trend_long and daily_bullish:
            new_signal = SIZE
        # Trigger 5: MACD cross with KAMA confirmation
        elif macd_bullish and kama_trend_long:
            new_signal = SIZE
        
        # SHORT ENTRY TRIGGERS (any one can trigger)
        # Trigger 1: KAMA cross short with daily resistance
        if kama_cross_short and (daily_bearish or rsi_bearish):
            new_signal = -SIZE
        # Trigger 2: KAMA trend + MACD bearish + RSI ok (trend continuation)
        elif kama_trend_short and macd_negative and rsi_bearish and price_below_kama:
            new_signal = -SIZE
        # Trigger 3: Daily bearish + KAMA trend + volume (regime + trend)
        elif daily_bearish and kama_trend_short and vol_confirm:
            new_signal = -SIZE
        # Trigger 4: RSI from overbought with KAMA resistance (mean reversion in trend)
        elif rsi_overbought and kama_trend_short and daily_bearish:
            new_signal = -SIZE
        # Trigger 5: MACD cross with KAMA confirmation
        elif macd_bearish and kama_trend_short:
            new_signal = -SIZE
        
        # Stoploss logic (Rule 6) - ATR based with trailing
        if position_side > 0 and entry_price > 0:
            # Update highest price since entry
            if close[i] > highest_since_entry:
                highest_since_entry = close[i]
            # Initial stoploss
            stop_loss = entry_price - 2.5 * atr[i]
            if close[i] < stop_loss:
                new_signal = 0.0  # Stoploss hit
            # Trail stop for longs (lock in profits)
            else:
                new_trailing = highest_since_entry - 2.5 * atr[i]
                if new_trailing > trailing_stop:
                    trailing_stop = new_trailing
                if close[i] < trailing_stop and trailing_stop > entry_price:
                    new_signal = 0.0
                # Take partial profit at 3R
                elif close[i] > entry_price + 3.0 * atr[entry_price > 0 and i > 0] if entry_price > 0 else 0:
                    pass  # Keep position, trail stop handles exit
        
        if position_side < 0 and entry_price > 0:
            # Update lowest price since entry
            if close[i] < lowest_since_entry or lowest_since_entry == 0:
                lowest_since_entry = close[i]
            # Initial stoploss
            stop_loss = entry_price + 2.5 * atr[i]
            if close[i] > stop_loss:
                new_signal = 0.0  # Stoploss hit
            # Trail stop for shorts (lock in profits)
            else:
                new_trailing = lowest_since_entry + 2.5 * atr[i]
                if new_trailing < trailing_stop or trailing_stop == 0:
                    trailing_stop = new_trailing
                if close[i] > trailing_stop and trailing_stop < entry_price:
                    new_signal = 0.0
        
        # Update position tracking
        if new_signal != 0 and position_side == 0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_since_entry = close[i] if position_side > 0 else 0.0
            lowest_since_entry = close[i] if position_side < 0 else 0.0
        elif new_signal != 0 and position_side != 0:
            if np.sign(new_signal) != position_side:
                entry_price = close[i]
                position_side = np.sign(new_signal)
                trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        elif new_signal == 0 and position_side != 0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_since_entry = 0.0
            lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals