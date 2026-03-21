#!/usr/bin/env python3
"""
Hypothesis: 1h primary with 4h HMA trend + MACD momentum + Bollinger entry timing
- 4h HMA provides stable trend direction (reduces 1h noise)
- MACD histogram confirms momentum in trend direction
- Bollinger Band position for entry timing (buy near lower band in uptrend)
- Volume spike confirms breakout validity
- ATR trailing stop for risk management
- Discrete position sizing to minimize fee churn
Timeframe: 1h (primary), 4h (HTF trend filter)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_hma_macd_bb_entry_1h_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while maintaining smoothness"""
    close_s = pd.Series(close)
    wma_half = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean().values
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean().values
    hull = 2 * wma_half - wma_full
    hma = pd.Series(hull).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean().values
    return hma

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_macd(close, fast=12, slow=26, signal=9):
    """MACD with histogram"""
    close_s = pd.Series(close)
    ema_fast = close_s.ewm(span=fast, min_periods=fast, adjust=False).mean().values
    ema_slow = close_s.ewm(span=slow, min_periods=slow, adjust=False).mean().values
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=signal, min_periods=signal, adjust=False).mean().values
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_bollinger(close, period=20, std_dev=2.0):
    """Bollinger Bands"""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    bb_pct = (close - lower) / (upper - lower + 1e-10)  # Position within bands (0-1)
    return sma, upper, lower, bb_pct

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    return rsi

def calculate_volume_ratio(volume, period=20):
    """Volume ratio vs rolling average"""
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    ratio = volume / (vol_sma + 1e-10)
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    # 4h HMA for trend direction
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_50 = calculate_hma(df_4h['close'].values, 50)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_50_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_50)
    
    # 1h indicators
    hma_1h_16 = calculate_hma(close, 16)
    hma_1h_48 = calculate_hma(close, 48)
    atr = calculate_atr(high, low, close, period=14)
    macd_line, macd_signal, macd_hist = calculate_macd(close)
    bb_sma, bb_upper, bb_lower, bb_pct = calculate_bollinger(close)
    rsi_1h = calculate_rsi(close, 14)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.30   # Long position size 30%
    SIZE_SHORT = 0.25  # Short position size 25% (asymmetric)
    SIZE_HALF = 0.15   # Half position for scaling
    
    prev_signal = 0.0
    entry_price = 0.0
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # HTF trend regime (4h HMA crossover)
        htf_bull = hma_4h_21_aligned[i] > hma_4h_50_aligned[i]
        htf_bear = hma_4h_21_aligned[i] < hma_4h_50_aligned[i]
        
        # 4h trend strength (slope)
        if i >= 4:
            hma_slope_4h = (hma_4h_21_aligned[i] - hma_4h_21_aligned[i-4]) / (hma_4h_21_aligned[i-4] + 1e-10)
        else:
            hma_slope_4h = 0.0
        
        # 1h trend
        trend_1h = 1.0 if hma_1h_16[i] > hma_1h_48[i] else -1.0
        
        # MACD momentum
        macd_bull = macd_hist[i] > 0 and macd_hist[i] > macd_hist[i-1] if i > 0 else False
        macd_bear = macd_hist[i] < 0 and macd_hist[i] < macd_hist[i-1] if i > 0 else False
        macd_cross_up = macd_hist[i] > 0 and macd_hist[i-1] <= 0 if i > 0 else False
        macd_cross_down = macd_hist[i] < 0 and macd_hist[i-1] >= 0 if i > 0 else False
        
        # Bollinger Band position
        bb_low = bb_pct[i] < 0.3  # Near lower band
        bb_high = bb_pct[i] > 0.7  # Near upper band
        bb_middle = bb_pct[i] > 0.4 and bb_pct[i] < 0.6  # Near middle
        
        # Volume confirmation
        vol_confirmed = vol_ratio[i] > 1.3
        
        # RSI filter
        rsi_oversold = rsi_1h[i] < 40
        rsi_overbought = rsi_1h[i] > 60
        rsi_neutral = rsi_1h[i] > 45 and rsi_1h[i] < 55
        
        # ATR stoploss level
        atr_stop = 2.5 * atr[i]
        
        # Check stoploss first (trailing stop)
        if position_side == 1:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - atr_stop
            if close[i] < trailing_stop or close[i] < entry_price - atr_stop:
                signals[i] = 0.0
                position_side = 0
                prev_signal = 0.0
                continue
        elif position_side == -1:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + atr_stop
            if close[i] > trailing_stop or close[i] > entry_price + atr_stop:
                signals[i] = 0.0
                position_side = 0
                prev_signal = 0.0
                continue
        
        # Entry logic - asymmetric based on HTF regime
        if htf_bull:  # Bull regime - prefer longs
            # Primary long entry: trend + MACD + BB pullback
            if trend_1h > 0 and macd_bull and bb_low:
                signals[i] = SIZE_LONG
                if prev_signal == 0:
                    position_side = 1
                    entry_price = close[i]
                    highest_since_entry = close[i]
            # MACD cross up confirmation
            elif trend_1h > 0 and macd_cross_up and vol_confirmed:
                signals[i] = SIZE_LONG
                if prev_signal == 0:
                    position_side = 1
                    entry_price = close[i]
                    highest_since_entry = close[i]
            # RSI pullback in uptrend
            elif trend_1h > 0 and rsi_oversold and bb_middle:
                signals[i] = SIZE_LONG * 0.8
                if prev_signal == 0:
                    position_side = 1
                    entry_price = close[i]
                    highest_since_entry = close[i]
            # Exit on overbought
            elif rsi_overbought and bb_high:
                signals[i] = 0.0
                position_side = 0
            else:
                signals[i] = prev_signal
                
        elif htf_bear:  # Bear regime - prefer shorts
            # Primary short entry: trend + MACD + BB rally
            if trend_1h < 0 and macd_bear and bb_high:
                signals[i] = -SIZE_SHORT
                if prev_signal == 0:
                    position_side = -1
                    entry_price = close[i]
                    lowest_since_entry = close[i]
            # MACD cross down confirmation
            elif trend_1h < 0 and macd_cross_down and vol_confirmed:
                signals[i] = -SIZE_SHORT
                if prev_signal == 0:
                    position_side = -1
                    entry_price = close[i]
                    lowest_since_entry = close[i]
            # RSI rally in downtrend
            elif trend_1h < 0 and rsi_overbought and bb_middle:
                signals[i] = -SIZE_SHORT * 0.8
                if prev_signal == 0:
                    position_side = -1
                    entry_price = close[i]
                    lowest_since_entry = close[i]
            # Exit on oversold
            elif rsi_oversold and bb_low:
                signals[i] = 0.0
                position_side = 0
            else:
                signals[i] = prev_signal
                
        else:  # Neutral regime - reduce positions or flat
            if position_side == 1:
                signals[i] = SIZE_HALF  # Reduce long
            elif position_side == -1:
                signals[i] = -SIZE_HALF  # Reduce short
            else:
                signals[i] = 0.0  # Stay flat
        
        # Discretize signal to reduce churn
        if abs(signals[i]) < 0.10:
            signals[i] = 0.0
        elif signals[i] > 0:
            if signals[i] >= 0.25:
                signals[i] = SIZE_LONG
            elif signals[i] >= 0.15:
                signals[i] = SIZE_HALF
            else:
                signals[i] = 0.0
        else:
            if signals[i] <= -0.20:
                signals[i] = -SIZE_SHORT
            elif signals[i] <= -0.10:
                signals[i] = -SIZE_HALF
            else:
                signals[i] = 0.0
        
        prev_signal = signals[i]
    
    return signals