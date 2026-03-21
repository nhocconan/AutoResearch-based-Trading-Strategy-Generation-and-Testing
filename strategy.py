#!/usr/bin/env python3
"""
EXPERIMENT #008 - MTF HMA Trend + RSI Pullback Strategy (30m)
==========================================================
Hypothesis: Combining 4h HMA trend filter with 30m RSI pullback entries will 
capture trends while avoiding chasing extended moves. The 4h HMA provides clean 
trend direction, while 30m RSI < 40 in uptrend (or > 60 in downtrend) gives 
optimal entry points with better risk/reward.

Key features:
- 4h HMA(21) for trend direction (loaded ONCE before loop via mtf_data)
- 30m RSI(14) for entry timing (pullback entries)
- ATR(14) trailing stoploss at 2*ATR
- Discrete position sizing: 0.0, ±0.30
- Take profit: reduce to half at 2R profit
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_hma_rsi_pullback_30m_v2"
timeframe = "30m"
leverage = 1.0


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=1, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=1, adjust=False).mean()
    diff = 2 * wma1 - wma2
    hma = diff.ewm(span=int(np.sqrt(period)), min_periods=1, adjust=False).mean()
    return hma.values


def calculate_rsi(close, period=14):
    """Calculate RSI with proper min_periods"""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values


def calculate_atr(high, low, close, period=14):
    """Calculate ATR with proper min_periods"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], 
                    abs(high[i] - close[i-1]), 
                    abs(low[i] - close[i-1]))
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    return atr


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (CRITICAL - Rule 1)
    df_4h = get_htf_data(prices, '4h')
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    rsi = calculate_rsi(close, 14)
    atr = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size - conservative for drawdown control
    
    entry_price = 0.0
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    tp_triggered = False
    
    for i in range(50, n):
        if np.isnan(hma_4h_aligned[i]) or np.isnan(rsi[i]) or np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        trend_4h = hma_4h_aligned[i]
        trend_bullish = close[i] > trend_4h
        trend_bearish = close[i] < trend_4h
        
        rsi_oversold = rsi[i] < 40
        rsi_overbought = rsi[i] > 60
        
        if position_side == 0:
            # Enter long: bullish trend + RSI pullback
            if trend_bullish and rsi_oversold:
                signals[i] = SIZE
                entry_price = close[i]
                position_side = 1
                highest_close = close[i]
                tp_triggered = False
            # Enter short: bearish trend + RSI bounce
            elif trend_bearish and rsi_overbought:
                signals[i] = -SIZE
                entry_price = close[i]
                position_side = -1
                lowest_close = close[i]
                tp_triggered = False
            else:
                signals[i] = 0.0
        
        elif position_side == 1:
            # Long position management
            highest_close = max(highest_close, close[i])
            profit = close[i] - entry_price
            profit_r = profit / atr[i] if atr[i] > 0 else 0
            
            # Take profit at 2R: reduce to half
            if profit_r >= 2.0 and not tp_triggered:
                signals[i] = SIZE / 2
                tp_triggered = True
            # Stoploss at -2R: close position
            elif profit_r <= -2.0:
                signals[i] = 0.0
                position_side = 0
                entry_price = 0.0
                tp_triggered = False
            # Trailing stop: exit if price drops 2*ATR from highest
            elif close[i] < highest_close - 2 * atr[i]:
                signals[i] = 0.0
                position_side = 0
                entry_price = 0.0
                tp_triggered = False
            # Trend reversal: exit if price crosses below 4h HMA
            elif not trend_bullish:
                signals[i] = 0.0
                position_side = 0
                entry_price = 0.0
                tp_triggered = False
            else:
                signals[i] = SIZE if not tp_triggered else SIZE / 2
        
        elif position_side == -1:
            # Short position management
            lowest_close = min(lowest_close, close[i])
            profit = entry_price - close[i]
            profit_r = profit / atr[i] if atr[i] > 0 else 0
            
            # Take profit at 2R: reduce to half
            if profit_r >= 2.0 and not tp_triggered:
                signals[i] = -SIZE / 2
                tp_triggered = True
            # Stoploss at -2R: close position
            elif profit_r <= -2.0:
                signals[i] = 0.0
                position_side = 0
                entry_price = 0.0
                tp_triggered = False
            # Trailing stop: exit if price rises 2*ATR from lowest
            elif close[i] > lowest_close + 2 * atr[i]:
                signals[i] = 0.0
                position_side = 0
                entry_price = 0.0
                tp_triggered = False
            # Trend reversal: exit if price crosses above 4h HMA
            elif not trend_bearish:
                signals[i] = 0.0
                position_side = 0
                entry_price = 0.0
                tp_triggered = False
            else:
                signals[i] = -SIZE if not tp_triggered else -SIZE / 2
    
    return signals