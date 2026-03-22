#!/usr/bin/env python3
"""
Hypothesis: 4h KAMA trend + 1d HTF regime filter + Bollinger BW detection
- KAMA adapts to volatility (better than HMA in range markets)
- 1d trend determines bull/bear regime (asymmetric entry thresholds)
- BB Width percentile detects squeeze vs expansion
- ATR trailing stop for risk management
- Fewer trades, larger moves only
Timeframe: 4h (primary), 1d (HTF trend filter)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_kama_bb_regime_4h_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=10, fast=2, slow=30):
    """Kaufman Adaptive Moving Average - adapts to market noise"""
    n = len(close)
    kama = np.zeros(n)
    kama[0] = close[0]
    
    # Efficiency Ratio
    for i in range(1, n):
        if i < period:
            kama[i] = close[i]
            continue
        
        change = abs(close[i] - close[i - period])
        volatility = np.sum(np.abs(np.diff(close[i-period:i+1])))
        
        if volatility == 0:
            er = 1.0
        else:
            er = change / volatility
        
        # Smoothed constants
        fast_sc = (2 / (fast + 1)) ** 2
        slow_sc = (2 / (slow + 1)) ** 2
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    return kama

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

def calculate_bb_width(close, period=20, std_mult=2.0):
    """Bollinger Band Width as % of price"""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    width = (upper - lower) / sma
    return width, sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA for trend filter
    close_1d_s = pd.Series(close_1d)
    ema_1d_21 = close_1d_s.ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_21)
    
    # 4h indicators
    kama = calculate_kama(close, period=10, fast=2, slow=30)
    kama_fast = calculate_kama(close, period=5, fast=2, slow=20)
    atr = calculate_atr(high, low, close, period=14)
    bb_width, bb_mid = calculate_bb_width(close, period=20, std_mult=2.0)
    
    # BB Width percentile (rolling 100 bars)
    bb_width_s = pd.Series(bb_width)
    bb_percentile = bb_width_s.rolling(window=100, min_periods=50).apply(
        lambda x: np.sum(x < x.iloc[-1]) / len(x), raw=False
    ).values
    bb_percentile = np.nan_to_num(bb_percentile, 0.5)
    
    # RSI calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=14, min_periods=14, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=14, min_periods=14, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size
    prev_signal = 0.0
    entry_price = 0.0
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # HTF trend regime (1d EMA)
        htf_trend = 1.0 if ema_1d_aligned[i] > 0 and close[i] > ema_1d_aligned[i] else -1.0
        
        # 4h trend (KAMA crossover)
        trend_4h = 1.0 if kama_fast[i] > kama[i] else -1.0
        
        # BB regime: <0.3 = squeeze (mean reversion), >0.7 = expansion (trend)
        bb_regime = bb_percentile[i]
        
        # ATR stoploss level
        atr_stop = 2.5 * atr[i]
        
        # Check stoploss first
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
        if htf_trend > 0:  # Bull regime - prefer longs
            if trend_4h > 0 and rsi[i] < 55 and bb_regime > 0.4:
                # Trend + not overbought + some volatility
                signals[i] = SIZE
                if prev_signal == 0:
                    position_side = 1
                    entry_price = close[i]
                    highest_since_entry = close[i]
            elif rsi[i] > 70:
                # Overbought - reduce or flat
                signals[i] = 0.0
                position_side = 0
            else:
                signals[i] = prev_signal
        else:  # Bear regime - prefer shorts or flat
            if trend_4h < 0 and rsi[i] > 45 and bb_regime > 0.4:
                # Downtrend + not oversold + some volatility
                signals[i] = -SIZE
                if prev_signal == 0:
                    position_side = -1
                    entry_price = close[i]
                    lowest_since_entry = close[i]
            elif rsi[i] < 30:
                # Oversold - cover shorts
                signals[i] = 0.0
                position_side = 0
            else:
                signals[i] = prev_signal
        
        prev_signal = signals[i]
    
    return signals