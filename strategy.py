#!/usr/bin/env python3
"""
Experiment #003: 1h Multi-Timeframe KAMA Trend + RSI Pullback + 4h HMA Filter
Hypothesis: 1h primary timeframe captures intermediate trends better than daily.
4h HMA provides major trend regime filter (bull/bear). 
KAMA (Kaufman Adaptive) adapts to volatility - faster in trends, slower in chop.
RSI pullback entries (RSI 40-60 zone) avoid chasing tops/bottoms.
Volume confirmation filters false breakouts.
ATR stoploss (2.5x) protects against crashes like 2022.
Position sizing 0.25-0.30 discrete levels to limit drawdown.
This should generate 30-60 trades/year with better risk-adjusted returns than EMA crossover.
Key improvement: KAMA adapts to market regime better than static EMA, reducing whipsaw.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_kama_rsi_4h_v1"
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

def calculate_kama(close, period=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average."""
    close_s = pd.Series(close)
    # Efficiency Ratio
    change = np.abs(close - np.roll(close, period))
    change[0:period] = np.abs(close[0:period] - close[0])
    volatility = np.zeros(len(close))
    for i in range(period, len(close)):
        volatility[i] = np.sum(np.abs(close[i-period+1:i+1] - np.roll(close[i-period+1:i+1], 1)))
    volatility[0:period] = change[0:period]
    er = np.where(volatility > 0, change / volatility, 0.0)
    # Smoothing constant
    sc = (er * (2.0/(fast+1) - 2.0/(slow+1)) + 2.0/(slow+1)) ** 2
    # KAMA calculation
    kama = np.zeros(len(close))
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return wma3.values

def calculate_sma(close, period):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

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
    
    # Load 12h HTF for additional regime filter
    df_12h = get_htf_data(prices, '12h')
    hma_12h = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    kama_fast = calculate_kama(close, 10, 2, 30)
    kama_slow = calculate_kama(close, 20, 2, 30)
    rsi = calculate_rsi(close, 14)
    sma_200 = calculate_sma(close, 200)
    
    # Volume SMA for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_sma = np.nan_to_num(vol_sma, nan=1.0)
    vol_sma[vol_sma == 0] = 1.0
    
    signals = np.zeros(n)
    SIZE = 0.28
    HALF_SIZE = 0.14
    
    # Track positions for stoploss
    entry_price = np.zeros(n)
    position_side = 0
    trailing_stop = np.zeros(n)
    
    for i in range(250, n):  # Start after 200 SMA is valid
        # 4h trend filter (major regime)
        hma_4h_valid = hma_4h_aligned[i] > 0 and not np.isnan(hma_4h_aligned[i])
        hma_12h_valid = hma_12h_aligned[i] > 0 and not np.isnan(hma_12h_aligned[i])
        
        trend_4h_bull = hma_4h_valid and close[i] > hma_4h_aligned[i]
        trend_4h_bear = hma_4h_valid and close[i] < hma_4h_aligned[i]
        
        trend_12h_bull = hma_12h_valid and close[i] > hma_12h_aligned[i]
        trend_12h_bear = hma_12h_valid and close[i] < hma_12h_aligned[i]
        
        # 1h KAMA crossover signals
        kama_cross_long = kama_fast[i] > kama_slow[i] and kama_fast[i-1] <= kama_slow[i-1]
        kama_cross_short = kama_fast[i] < kama_slow[i] and kama_fast[i-1] >= kama_slow[i-1]
        
        # KAMA trend alignment
        kama_trend_long = kama_fast[i] > kama_slow[i] and close[i] > kama_fast[i]
        kama_trend_short = kama_fast[i] < kama_slow[i] and close[i] < kama_fast[i]
        
        # RSI pullback zone (not extreme - avoids chasing)
        rsi_long = rsi[i] > 35 and rsi[i] < 65
        rsi_short = rsi[i] > 35 and rsi[i] < 65
        rsi_rising = rsi[i] > rsi[i-2] if i > 2 else True
        rsi_falling = rsi[i] < rsi[i-2] if i > 2 else True
        
        # Price vs SMA200 filter
        above_sma200 = close[i] > sma_200[i] if not np.isnan(sma_200[i]) else True
        below_sma200 = close[i] < sma_200[i] if not np.isnan(sma_200[i]) else True
        
        # Volume confirmation
        vol_ratio = volume[i] / vol_sma[i] if vol_sma[i] > 0 else 1.0
        vol_confirm = vol_ratio > 0.6  # Relaxed volume filter
        
        # Entry logic - designed to generate sufficient trades
        new_signal = 0.0
        
        # Long entry: 4h bullish + KAMA trend + RSI ok + volume
        if trend_4h_bull and kama_trend_long and rsi_long and vol_confirm:
            new_signal = SIZE
        # Long on KAMA crossover with 4h support
        elif trend_4h_bull and kama_cross_long and rsi[i] > 30:
            new_signal = SIZE
        # Long on pullback in uptrend (4h and 12h agree)
        elif trend_4h_bull and trend_12h_bull and kama_trend_long and rsi[i] > 40:
            new_signal = SIZE
        
        # Short entry: 4h bearish + KAMA trend + RSI ok
        if new_signal == 0:  # Only short if not already long
            if trend_4h_bear and kama_trend_short and rsi_short and vol_confirm:
                new_signal = -SIZE
            # Short on KAMA crossover with 4h resistance
            elif trend_4h_bear and kama_cross_short and rsi[i] < 70:
                new_signal = -SIZE
            # Short on rally in downtrend (4h and 12h agree)
            elif trend_4h_bear and trend_12h_bear and kama_trend_short and rsi[i] < 60:
                new_signal = -SIZE
        
        # Stoploss logic (Rule 6) - ATR based
        if position_side > 0 and entry_price[i-1] > 0:
            stop_loss = entry_price[i-1] - 2.5 * atr[i]
            if close[i] < stop_loss:
                new_signal = 0.0  # Stoploss hit
            # Trail stop for longs - lock in profits
            elif close[i] > entry_price[i-1] + 2.0 * atr[i]:
                trailing_stop[i] = max(trailing_stop[i-1] if i > 0 else 0, close[i] - 1.5 * atr[i])
                if close[i] < trailing_stop[i] and new_signal == 0:
                    new_signal = HALF_SIZE
        
        if position_side < 0 and entry_price[i-1] > 0:
            stop_loss = entry_price[i-1] + 2.5 * atr[i]
            if close[i] > stop_loss:
                new_signal = 0.0  # Stoploss hit
            # Trail stop for shorts - lock in profits
            elif close[i] < entry_price[i-1] - 2.0 * atr[i]:
                trailing_stop[i] = min(trailing_stop[i-1] if i > 0 else 0, close[i] + 1.5 * atr[i])
                if close[i] > trailing_stop[i] and new_signal == 0:
                    new_signal = -HALF_SIZE
        
        # Update position tracking
        if new_signal != 0 and position_side == 0:
            entry_price[i] = close[i]
            position_side = np.sign(new_signal)
            trailing_stop[i] = entry_price[i] - 2.5 * atr[i] if position_side > 0 else entry_price[i] + 2.5 * atr[i]
        elif new_signal != 0 and position_side != 0:
            if np.sign(new_signal) != position_side:
                entry_price[i] = close[i]
                position_side = np.sign(new_signal)
                trailing_stop[i] = entry_price[i] - 2.5 * atr[i] if position_side > 0 else entry_price[i] + 2.5 * atr[i]
            else:
                entry_price[i] = entry_price[i-1] if i > 0 else close[i]
                trailing_stop[i] = trailing_stop[i-1] if i > 0 else entry_price[i]
        else:
            entry_price[i] = entry_price[i-1] if i > 0 else 0
            trailing_stop[i] = trailing_stop[i-1] if i > 0 else 0
            if position_side != 0 and new_signal == 0:
                position_side = 0  # Position closed
        
        signals[i] = new_signal
    
    return signals