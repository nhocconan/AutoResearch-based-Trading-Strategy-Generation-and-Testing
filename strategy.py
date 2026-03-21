#!/usr/bin/env python3
"""
Experiment #018: 1d Daily Trend + Weekly Regime + Volume Breakout
Hypothesis: Daily timeframe captures multi-week swings with cleaner signals than intraday.
Weekly HMA provides major bull/bear regime filter to avoid counter-trend trades.
Daily HMA crossover + RSI momentum + volume confirmation for entries.
ATR-based stoploss (2.5x) and trailing stop protect against crashes.
Relaxed RSI thresholds (25-75) ensure sufficient trades on daily TF.
Position sizing at 0.30 with discrete levels to minimize fee churn.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_hma_weekly_regime_vol_v1"
timeframe = "1d"
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
    
    # Load weekly HTF data ONCE before loop (Rule 1)
    df_1w = get_htf_data(prices, '1w')
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate daily indicators
    atr = calculate_atr(high, low, close, 14)
    hma_fast = calculate_hma(close, 12)
    hma_slow = calculate_hma(close, 36)
    rsi = calculate_rsi(close, 14)
    macd_line, macd_signal, macd_hist = calculate_macd(close, 12, 26, 9)
    
    # Volume SMA for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_sma = np.nan_to_num(vol_sma, nan=np.mean(volume))
    
    # Price momentum (ROC)
    roc = np.zeros(n)
    roc[10:] = (close[10:] - close[:-10]) / close[:-10] * 100
    
    signals = np.zeros(n)
    SIZE = 0.30
    HALF_SIZE = 0.15
    
    # Track positions for stoploss
    entry_price = np.zeros(n)
    position_side = 0
    trailing_stop = np.zeros(n)
    max_close = np.zeros(n)
    min_close = np.zeros(n)
    
    for i in range(50, n):
        # Weekly trend filter (major regime)
        weekly_bullish = hma_1w_aligned[i] > 0 and close[i] > hma_1w_aligned[i]
        weekly_bearish = hma_1w_aligned[i] > 0 and close[i] < hma_1w_aligned[i]
        
        # Daily HMA trend
        hma_trend_long = hma_fast[i] > hma_slow[i]
        hma_trend_short = hma_fast[i] < hma_slow[i]
        
        # HMA crossover signals
        hma_cross_long = hma_fast[i] > hma_slow[i] and hma_fast[i-1] <= hma_slow[i-1]
        hma_cross_short = hma_fast[i] < hma_slow[i] and hma_fast[i-1] >= hma_slow[i-1]
        
        # MACD confirmation
        macd_bullish = macd_hist[i] > 0 and macd_hist[i-1] <= 0
        macd_bearish = macd_hist[i] < 0 and macd_hist[i-1] >= 0
        macd_positive = macd_hist[i] > 0
        macd_negative = macd_hist[i] < 0
        
        # RSI momentum (relaxed for more trades on daily TF)
        rsi_long = rsi[i] > 40 and rsi[i] < 75
        rsi_short = rsi[i] > 25 and rsi[i] < 60
        rsi_rising = rsi[i] > rsi[i-3] if i > 3 else True
        rsi_falling = rsi[i] < rsi[i-3] if i > 3 else True
        
        # Volume confirmation
        vol_confirm_long = volume[i] > vol_sma[i] * 0.8 if vol_sma[i] > 0 else True
        vol_confirm_short = volume[i] > vol_sma[i] * 0.8 if vol_sma[i] > 0 else True
        
        # Momentum confirmation
        mom_long = roc[i] > -2.0
        mom_short = roc[i] < 2.0
        
        # Entry logic - relaxed conditions to ensure ≥10 trades
        new_signal = 0.0
        
        # Long entry: weekly bullish + daily trend + momentum
        if weekly_bullish and hma_trend_long and rsi_long and vol_confirm_long:
            new_signal = SIZE
        # Long on HMA crossover with weekly support
        elif weekly_bullish and hma_cross_long and rsi[i] > 35:
            new_signal = SIZE
        # Long on MACD flip with volume
        elif weekly_bullish and macd_bullish and vol_confirm_long:
            new_signal = SIZE
        # Long on pure momentum break (ensure trades)
        elif weekly_bullish and hma_trend_long and roc[i] > 3.0:
            new_signal = SIZE
        
        # Short entry: weekly bearish + daily trend + momentum
        elif weekly_bearish and hma_trend_short and rsi_short and vol_confirm_short:
            new_signal = -SIZE
        # Short on HMA crossover with weekly resistance
        elif weekly_bearish and hma_cross_short and rsi[i] < 65:
            new_signal = -SIZE
        # Short on MACD flip with volume
        elif weekly_bearish and macd_bearish and vol_confirm_short:
            new_signal = -SIZE
        # Short on pure momentum break (ensure trades)
        elif weekly_bearish and hma_trend_short and roc[i] < -3.0:
            new_signal = -SIZE
        
        # Stoploss logic (Rule 6) - ATR based
        if position_side > 0 and entry_price[i-1] > 0:
            stop_loss = entry_price[i-1] - 2.5 * atr[i]
            if close[i] < stop_loss:
                new_signal = 0.0  # Stoploss hit
            # Trail stop for longs
            else:
                max_close[i] = max(max_close[i-1] if i > 0 else close[i], close[i])
                trailing_stop[i] = max_close[i] - 2.5 * atr[i]
                if close[i] < trailing_stop[i] and trailing_stop[i] > 0:
                    new_signal = 0.0
                # Take partial profit at 3R
                elif close[i] > entry_price[i-1] + 3.0 * atr[i] and new_signal == SIZE:
                    new_signal = HALF_SIZE
        
        if position_side < 0 and entry_price[i-1] > 0:
            stop_loss = entry_price[i-1] + 2.5 * atr[i]
            if close[i] > stop_loss:
                new_signal = 0.0  # Stoploss hit
            # Trail stop for shorts
            else:
                min_close[i] = min(min_close[i-1] if i > 0 else close[i], close[i])
                trailing_stop[i] = min_close[i] + 2.5 * atr[i]
                if close[i] > trailing_stop[i] and trailing_stop[i] < 999999:
                    new_signal = 0.0
                # Take partial profit at 3R
                elif close[i] < entry_price[i-1] - 3.0 * atr[i] and new_signal == -SIZE:
                    new_signal = -HALF_SIZE
        
        # Update position tracking
        if new_signal != 0 and position_side == 0:
            entry_price[i] = close[i]
            position_side = np.sign(new_signal)
            max_close[i] = close[i]
            min_close[i] = close[i]
            trailing_stop[i] = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
        elif new_signal != 0 and position_side != 0:
            if np.sign(new_signal) != position_side:
                entry_price[i] = close[i]
                position_side = np.sign(new_signal)
                max_close[i] = close[i]
                min_close[i] = close[i]
                trailing_stop[i] = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            else:
                max_close[i] = max_close[i-1] if i > 0 else close[i]
                min_close[i] = min_close[i-1] if i > 0 else close[i]
                trailing_stop[i] = trailing_stop[i-1] if i > 0 else trailing_stop[i]
        else:
            entry_price[i] = entry_price[i-1] if i > 0 else 0
            max_close[i] = max_close[i-1] if i > 0 else close[i]
            min_close[i] = min_close[i-1] if i > 0 else close[i]
            trailing_stop[i] = trailing_stop[i-1] if i > 0 else 0
            if position_side != 0 and new_signal == 0:
                position_side = 0  # Position closed
        
        signals[i] = new_signal
    
    return signals