#!/usr/bin/env python3
"""
Hypothesis: 1h primary with 4h HMA trend filter + 1h EMA/RSI entry + ATR stoploss
- 4h HMA(21) slope determines macro trend (call ONCE before loop)
- 1h EMA(21) + RSI(14) for entry timing on pullbacks
- ATR(14) trailing stoploss at 2.5*ATR
- Discrete signals: 0.0, ±0.25, ±0.30 to minimize fee churn
- Looser entry conditions to ensure ≥10 trades per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_hma_ema_rsi_1h_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    def wma(arr, n):
        weights = np.arange(1, n + 1)
        result = np.zeros(len(arr))
        for i in range(n - 1, len(arr)):
            result[i] = np.sum(arr[i - n + 1:i + 1] * weights) / np.sum(weights)
        return result
    
    wma1 = wma(close, period // 2)
    wma2 = wma(close, period)
    diff = 2 * wma1 - wma2
    hma = wma(diff, int(np.sqrt(period)))
    return hma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 4h data ONCE before loop (CRITICAL - Rule 1)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 4h HMA(21) for trend
    hma_4h_raw = calculate_hma(close_4h, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate 1h indicators
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
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
    
    # EMA(21) for dynamic support/resistance
    ema21 = close_s.ewm(span=21, min_periods=21, adjust=False).mean().values
    
    # EMA(50) for trend confirmation
    ema50 = close_s.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    signals = np.zeros(n)
    SIZE = 0.30
    HALF_SIZE = 0.15
    
    # Track positions for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(60, n):
        # Get aligned 4h HMA for trend
        hma_4h = hma_4h_aligned[i]
        hma_4h_prev = hma_4h_aligned[i-1] if i > 0 else hma_4h
        
        # Determine 4h trend direction from HMA slope
        hma_slope = hma_4h - hma_4h_prev
        trend_bullish = hma_slope > 0 and close[i] > hma_4h
        trend_bearish = hma_slope < 0 and close[i] < hma_4h
        
        # 1h trend confirmation
        ema_trend_bull = ema21[i] > ema50[i]
        ema_trend_bear = ema21[i] < ema50[i]
        
        prev_signal = signals[i-1]
        
        # LONG entry: 4h bullish + 1h EMA bullish + RSI pullback (not overbought)
        if trend_bullish and ema_trend_bull:
            if rsi[i] >= 35 and rsi[i] <= 60:
                if not in_position or position_side <= 0:
                    signals[i] = SIZE
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    highest_since_entry = close[i]
                else:
                    signals[i] = SIZE
                    highest_since_entry = max(highest_since_entry, close[i])
            elif rsi[i] > 60:
                # Overbought - hold but reduce
                if in_position and position_side == 1:
                    signals[i] = HALF_SIZE
                    highest_since_entry = max(highest_since_entry, close[i])
        
        # SHORT entry: 4h bearish + 1h EMA bearish + RSI pullback (not oversold)
        elif trend_bearish and ema_trend_bear:
            if rsi[i] >= 40 and rsi[i] <= 65:
                if not in_position or position_side >= 0:
                    signals[i] = -SIZE
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    lowest_since_entry = close[i]
                else:
                    signals[i] = -SIZE
                    lowest_since_entry = min(lowest_since_entry, close[i])
            elif rsi[i] < 40:
                # Oversold - hold but reduce
                if in_position and position_side == -1:
                    signals[i] = -HALF_SIZE
                    lowest_since_entry = min(lowest_since_entry, close[i])
        
        # Stoploss logic - trailing stop based on ATR
        if in_position and position_side == 1:
            highest_since_entry = max(highest_since_entry, close[i])
            trail_stop = highest_since_entry - 2.5 * atr[i]
            if close[i] < trail_stop:
                signals[i] = 0.0
                in_position = False
                position_side = 0
        
        if in_position and position_side == -1:
            lowest_since_entry = min(lowest_since_entry, close[i])
            trail_stop = lowest_since_entry + 2.5 * atr[i]
            if close[i] > trail_stop:
                signals[i] = 0.0
                in_position = False
                position_side = 0
        
        # Flat when no trend signal
        if signals[i] == 0.0 and prev_signal != 0.0:
            in_position = False
            position_side = 0
    
    return signals