#!/usr/bin/env python3
"""
Hypothesis: Daily timeframe with weekly trend filter reduces noise and improves risk-adjusted returns.
Using 1w HTF trend direction to filter 1d EMA crossover entries. ATR stoploss for risk control.
This should work across BTC/ETH/SOL as all follow multi-year trends.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_ema_rsi_atr_1d_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load weekly HTF data ONCE before loop (Rule 1)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly trend (HMA21 on weekly close)
    close_1w = df_1w['close'].values
    hma_1w = calculate_hma(close_1w, 21)
    
    # Align weekly trend to daily (auto shift(1) for completed bars only - Rule 2)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Daily indicators
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # EMA crossover (12/26)
    ema12 = close_s.ewm(span=12, min_periods=12, adjust=False).mean().values
    ema26 = close_s.ewm(span=26, min_periods=26, adjust=False).mean().values
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=14, min_periods=14, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=14, min_periods=14, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    
    # ATR(14) for stoploss
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Initialize signals and tracking
    signals = np.zeros(n)
    SIZE = 0.30
    HALF_SIZE = 0.15
    
    # Track position for stoploss
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(30, n):
        weekly_trend = hma_1w_aligned[i]
        weekly_bullish = close[i] > weekly_trend if not np.isnan(weekly_trend) else True
        
        ema_bullish = ema12[i] > ema26[i]
        ema_bearish = ema12[i] < ema26[i]
        
        # Check stoploss first (Rule 6)
        if position_side == 1:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * atr[i]
            if close[i] < trailing_stop or close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position_side = 0
                entry_price = 0.0
                continue
        elif position_side == -1:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * atr[i]
            if close[i] > trailing_stop or close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position_side = 0
                entry_price = 0.0
                continue
        
        # Entry logic with weekly filter
        if position_side == 0:
            # Long entry: weekly bullish + EMA bullish + RSI not overbought
            if weekly_bullish and ema_bullish and rsi[i] < 70:
                signals[i] = SIZE
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
            # Short entry: weekly bearish + EMA bearish + RSI not oversold
            elif not weekly_bullish and ema_bearish and rsi[i] > 30:
                signals[i] = -SIZE
                position_side = -1
                entry_price = close[i]
                lowest_since_entry = low[i]
            else:
                signals[i] = 0.0
        elif position_side == 1:
            # Hold long or exit on EMA cross
            if ema_bearish:
                signals[i] = 0.0
                position_side = 0
                entry_price = 0.0
            else:
                signals[i] = SIZE
        elif position_side == -1:
            # Hold short or exit on EMA cross
            if ema_bullish:
                signals[i] = 0.0
                position_side = 0
                entry_price = 0.0
            else:
                signals[i] = -SIZE
    
    return signals

def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma_half = close_s.ewm(span=period // 2, min_periods=period // 2, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    hma = (2 * wma_half - wma_full).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values