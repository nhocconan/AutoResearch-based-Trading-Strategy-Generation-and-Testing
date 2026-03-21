#!/usr/bin/env python3
"""
EXPERIMENT #005: 12h KAMA trend + 1d HMA filter + RSI pullback + ATR stoploss
Hypothesis: 12h primary with 1d HTF trend filter captures medium-term swings
while avoiding whipsaw. KAMA adapts to volatility better than EMA.
RSI pullback entries in direction of HTF trend improve win rate.
ATR-based stoploss limits drawdown. Discrete sizing reduces fee churn.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_kama_rsi_12h_v1"
timeframe = "12h"
leverage = 1.0

def calculate_kama(close, period=10, fast=2, slow=30):
    """Kaufman Adaptive Moving Average - adapts to market noise"""
    n = len(close)
    kama = np.zeros(n)
    kama[0] = close[0]
    
    for i in range(1, n):
        # Efficiency Ratio
        if i >= period:
            signal = abs(close[i] - close[i - period])
            noise = np.sum(np.abs(np.diff(close[i-period:i+1])))
            er = signal / noise if noise > 0 else 0
        else:
            er = 0
        
        # Smoothing constants
        fast_sc = (2 / (fast + 1)) ** 2
        slow_sc = (2 / (slow + 1)) ** 2
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    return kama

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility and stoploss"""
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

def calculate_hma(close, period=21):
    """Hull Moving Average for HTF trend"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    hma = (2 * wma1 - wma2).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    kama_fast = calculate_kama(close, period=10, fast=2, slow=30)
    kama_slow = calculate_kama(close, period=30, fast=2, slow=30)
    rsi = calculate_rsi(close, 14)
    atr = calculate_atr(high, low, close, 14)
    
    # Signal parameters
    SIZE_ENTRY = 0.30
    SIZE_HALF = 0.15
    STOPLOSS_MULT = 2.0
    TAKEPROFIT_MULT = 2.0
    
    signals = np.zeros(n)
    entry_price = np.zeros(n)
    position_side = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(50, n):
        # HTF trend from 1d HMA
        hma_trend = 1 if hma_1d_aligned[i] > hma_1d_aligned[i-1] else -1
        
        # 12h KAMA crossover
        kama_bull = kama_fast[i] > kama_slow[i]
        kama_bear = kama_fast[i] < kama_slow[i]
        
        # RSI pullback conditions
        rsi_not_overbought = rsi[i] < 70
        rsi_not_oversold = rsi[i] > 30
        rsi_pullback_long = rsi[i] < 55  # pullback in uptrend
        rsi_pullback_short = rsi[i] > 45  # pullback in downtrend
        
        # ATR volatility filter (avoid extremely low vol)
        atr_pct = atr[i] / close[i] * 100
        vol_ok = atr_pct > 0.5
        
        current_signal = 0.0
        
        # Entry logic
        if hma_trend > 0 and kama_bull and rsi_pullback_long and vol_ok:
            if position_side[i-1] <= 0:
                current_signal = SIZE_ENTRY
                entry_price[i] = close[i]
                position_side[i] = 1
                highest_since_entry[i] = close[i]
                lowest_since_entry[i] = close[i]
            else:
                current_signal = signals[i-1]
                position_side[i] = position_side[i-1]
                entry_price[i] = entry_price[i-1]
        elif hma_trend < 0 and kama_bear and rsi_pullback_short and vol_ok:
            if position_side[i-1] >= 0:
                current_signal = -SIZE_ENTRY
                entry_price[i] = close[i]
                position_side[i] = -1
                highest_since_entry[i] = close[i]
                lowest_since_entry[i] = close[i]
            else:
                current_signal = signals[i-1]
                position_side[i] = position_side[i-1]
                entry_price[i] = entry_price[i-1]
        else:
            # Check for exit conditions
            if position_side[i-1] != 0:
                # Update extremes
                if position_side[i-1] > 0:
                    highest_since_entry[i] = max(highest_since_entry[i-1], close[i])
                    lowest_since_entry[i] = lowest_since_entry[i-1]
                else:
                    lowest_since_entry[i] = min(lowest_since_entry[i-1], close[i])
                    highest_since_entry[i] = highest_since_entry[i-1]
                
                entry_price[i] = entry_price[i-1]
                position_side[i] = position_side[i-1]
                
                # Stoploss check
                if position_side[i-1] > 0:
                    stop_price = entry_price[i] - STOPLOSS_MULT * atr[i]
                    if close[i] < stop_price:
                        current_signal = 0.0
                        position_side[i] = 0
                    elif close[i] >= entry_price[i] + TAKEPROFIT_MULT * atr[i]:
                        # Take profit - reduce to half
                        current_signal = SIZE_HALF
                    else:
                        current_signal = signals[i-1]
                else:
                    stop_price = entry_price[i] + STOPLOSS_MULT * atr[i]
                    if close[i] > stop_price:
                        current_signal = 0.0
                        position_side[i] = 0
                    elif close[i] <= entry_price[i] - TAKEPROFIT_MULT * atr[i]:
                        # Take profit - reduce to half
                        current_signal = -SIZE_HALF
                    else:
                        current_signal = signals[i-1]
                
                # Exit if trend reverses
                if position_side[i-1] > 0 and hma_trend < 0 and kama_bear:
                    current_signal = 0.0
                    position_side[i] = 0
                elif position_side[i-1] < 0 and hma_trend > 0 and kama_bull:
                    current_signal = 0.0
                    position_side[i] = 0
            else:
                position_side[i] = 0
                entry_price[i] = 0
                highest_since_entry[i] = 0
                lowest_since_entry[i] = 0
        
        # Carry forward if no change
        if current_signal == 0.0 and i > 0:
            if position_side[i] == 0:
                signals[i] = 0.0
            else:
                signals[i] = signals[i-1]
        else:
            signals[i] = current_signal
    
    return signals