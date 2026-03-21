#!/usr/bin/env python3
"""
Experiment #027: 1h KAMA Adaptive Trend + 4h HMA Regime + RSI Momentum
Hypothesis: 1h timeframe captures multi-day swings while reducing noise vs 15m/30m.
KAMA (Kaufman Adaptive Moving Average) adapts to market volatility - fast in trends, slow in ranges.
4h HMA provides regime filter (avoid counter-trend trades in bear markets).
RSI momentum ensures we enter with momentum, not against prevailing trend.
Multiple entry triggers (KAMA cross, RSI cross, trend continuation) ensure ≥10 trades.
Position sizing 0.28 with 2.5x ATR stoploss protects against 2022-style crashes.
Volume confirmation filters false breakouts in low-liquidity periods.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_kama_4h_hma_rsi_v1"
timeframe = "1h"
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

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market noise - moves fast in trends, slow in ranges.
    Efficiency Ratio (ER) measures trend vs noise.
    """
    close_s = pd.Series(close)
    
    # Calculate price change and volatility
    price_change = np.abs(close - np.roll(close, period))
    price_change[:period] = np.abs(close[:period] - close[0])
    
    volatility = np.zeros(len(close))
    for i in range(period, len(close)):
        volatility[i] = np.sum(np.abs(close[i-period+1:i+1] - np.roll(close[i-period+1:i+1], 1)))
    volatility[:period] = price_change[:period]
    
    # Efficiency Ratio (ER) - 0 = noise, 1 = pure trend
    er = np.zeros(len(close))
    er[volatility > 0] = price_change[volatility > 0] / volatility[volatility > 0]
    er = np.clip(er, 0, 1)
    
    # Smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = er * (fast_sc - slow_sc) + slow_sc
    sc = np.nan_to_num(sc, nan=slow_sc)
    
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

def calculate_sma(values, period):
    """Calculate Simple Moving Average."""
    return pd.Series(values).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    
    # KAMA for adaptive trend following
    kama_fast = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    kama_slow = calculate_kama(close, period=20, fast_period=2, slow_period=30)
    
    # HMA for trend confirmation
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    
    # Volume SMA for confirmation
    vol_sma = calculate_sma(volume, 20)
    vol_sma = np.nan_to_num(vol_sma, nan=np.nanmean(volume))
    
    # Price momentum (ROC)
    roc_10 = np.zeros(n)
    roc_10[10:] = (close[10:] - close[:-10]) / close[:-10] * 100
    
    signals = np.zeros(n)
    SIZE = 0.28
    HALF_SIZE = 0.14
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    
    for i in range(100, n):
        # 4h trend filter (major regime)
        hma_4h_valid = hma_4h_aligned[i] > 0 and not np.isnan(hma_4h_aligned[i])
        fourh_bullish = hma_4h_valid and close[i] > hma_4h_aligned[i]
        fourh_bearish = hma_4h_valid and close[i] < hma_4h_aligned[i]
        
        # KAMA trend signals
        kama_bullish = kama_fast[i] > kama_slow[i]
        kama_bearish = kama_fast[i] < kama_slow[i]
        
        # KAMA cross signals (strongest entry trigger)
        kama_cross_long = kama_fast[i] > kama_slow[i] and kama_fast[i-1] <= kama_slow[i-1]
        kama_cross_short = kama_fast[i] < kama_slow[i] and kama_fast[i-1] >= kama_slow[i-1]
        
        # HMA trend confirmation
        hma_trend_long = hma_21[i] > hma_50[i] and not np.isnan(hma_21[i]) and not np.isnan(hma_50[i])
        hma_trend_short = hma_21[i] < hma_50[i] and not np.isnan(hma_21[i]) and not np.isnan(hma_50[i])
        
        # RSI momentum (relaxed thresholds for more trades)
        rsi_bullish = rsi[i] > 45 and rsi[i] < 75
        rsi_bearish = rsi[i] > 25 and rsi[i] < 55
        rsi_rising = rsi[i] > rsi[i-3] if i > 3 else True
        rsi_falling = rsi[i] < rsi[i-3] if i > 3 else True
        
        # Volume confirmation
        vol_confirm = volume[i] > vol_sma[i] * 0.9 if vol_sma[i] > 0 else True
        
        # Price above/below HMA21 for trend confirmation
        price_above_hma = close[i] > hma_21[i] if not np.isnan(hma_21[i]) else False
        price_below_hma = close[i] < hma_21[i] if not np.isnan(hma_21[i]) else False
        
        # ROC momentum
        roc_positive = roc_10[i] > 0.5
        roc_negative = roc_10[i] < -0.5
        
        # Entry logic - MULTIPLE triggers to ensure trades (Rule 9)
        new_signal = 0.0
        
        # LONG ENTRY TRIGGERS (any one can trigger)
        # Trigger 1: KAMA cross long with 4h support
        if kama_cross_long and (fourh_bullish or rsi_bullish):
            new_signal = SIZE
        # Trigger 2: KAMA bullish + HMA trend + RSI ok (trend continuation)
        elif kama_bullish and hma_trend_long and rsi_bullish and price_above_hma:
            new_signal = SIZE
        # Trigger 3: 4h bullish + KAMA bullish + volume (regime + trend)
        elif fourh_bullish and kama_bullish and vol_confirm:
            new_signal = SIZE
        # Trigger 4: RSI rising from neutral with KAMA support
        elif rsi_rising and rsi[i] > 50 and kama_bullish and roc_positive:
            new_signal = SIZE
        # Trigger 5: HMA cross with KAMA confirmation
        elif hma_21[i] > hma_50[i] and hma_21[i-1] <= hma_50[i-1] and kama_bullish:
            new_signal = SIZE
        
        # SHORT ENTRY TRIGGERS (any one can trigger)
        # Trigger 1: KAMA cross short with 4h resistance
        if kama_cross_short and (fourh_bearish or rsi_bearish):
            new_signal = -SIZE
        # Trigger 2: KAMA bearish + HMA trend + RSI ok (trend continuation)
        elif kama_bearish and hma_trend_short and rsi_bearish and price_below_hma:
            new_signal = -SIZE
        # Trigger 3: 4h bearish + KAMA bearish + volume (regime + trend)
        elif fourh_bearish and kama_bearish and vol_confirm:
            new_signal = -SIZE
        # Trigger 4: RSI falling from neutral with KAMA support
        elif rsi_falling and rsi[i] < 50 and kama_bearish and roc_negative:
            new_signal = -SIZE
        # Trigger 5: HMA cross with KAMA confirmation
        elif hma_21[i] < hma_50[i] and hma_21[i-1] >= hma_50[i-1] and kama_bearish:
            new_signal = -SIZE
        
        # Stoploss logic (Rule 6) - ATR based with trailing
        if position_side > 0 and entry_price > 0:
            stop_loss = entry_price - 2.5 * atr[i]
            if close[i] < stop_loss:
                new_signal = 0.0  # Stoploss hit
            # Trail stop for longs
            else:
                new_trailing = close[i] - 2.5 * atr[i]
                if new_trailing > trailing_stop:
                    trailing_stop = new_trailing
                if close[i] < trailing_stop and trailing_stop > stop_loss:
                    new_signal = 0.0
                # Take partial profit at 3R
                elif close[i] > entry_price + 3.0 * atr[i] and signals[i-1] == SIZE:
                    new_signal = HALF_SIZE
        
        if position_side < 0 and entry_price > 0:
            stop_loss = entry_price + 2.5 * atr[i]
            if close[i] > stop_loss:
                new_signal = 0.0  # Stoploss hit
            # Trail stop for shorts
            else:
                new_trailing = close[i] + 2.5 * atr[i]
                if new_trailing < trailing_stop or trailing_stop == 0:
                    trailing_stop = new_trailing
                if close[i] > trailing_stop and trailing_stop < stop_loss:
                    new_signal = 0.0
                # Take partial profit at 3R
                elif close[i] < entry_price - 3.0 * atr[i] and signals[i-1] == -SIZE:
                    new_signal = -HALF_SIZE
        
        # Update position tracking
        if new_signal != 0 and position_side == 0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
        elif new_signal != 0 and position_side != 0:
            if np.sign(new_signal) != position_side:
                entry_price = close[i]
                position_side = np.sign(new_signal)
                trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
        elif new_signal == 0 and position_side != 0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
        
        signals[i] = new_signal
    
    return signals