#!/usr/bin/env python3
"""
EXPERIMENT #002 - Multi-Timeframe EMA Trend + RSI Pullback Strategy
====================================================================
Hypothesis: Using 1d EMA(21/55) crossover as trend filter with 4h RSI pullback 
entries will provide cleaner trend following with better entry timing. The daily 
trend filter reduces whipsaws, while 4h RSI allows entering on pullbacks within 
the trend direction.

Key features:
- 1d EMA trend filter (only trade in direction of daily trend)
- 4h RSI(14) for pullback entries (RSI<40 for longs, RSI>60 for shorts)
- ATR(14) trailing stoploss
- Discrete position sizing (0.0, ±0.20, ±0.35)
- Take profit at 2R (reduce to half position)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_ema_rsi_pullback_4h_v1"
timeframe = "4h"
leverage = 1.0


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (CRITICAL - Rule 1)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily EMA(21) and EMA(55)
    close_1d = df_1d['close'].values
    ema21_1d = pd.Series(close_1d).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema55_1d = pd.Series(close_1d).ewm(span=55, min_periods=55, adjust=False).mean().values
    
    # Align daily indicators to 4h timeframe (auto shift(1) for completed bars - Rule 2)
    ema21_aligned = align_htf_to_ltf(prices, df_1d, ema21_1d)
    ema55_aligned = align_htf_to_ltf(prices, df_1d, ema55_1d)
    
    # Calculate 4h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 4h ATR(14) for stoploss
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], 
                    abs(high[i] - close[i-1]), 
                    abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Generate signals with state tracking
    signals = np.zeros(n)
    SIZE_FULL = 0.35  # 35% position size - critical for drawdown control
    SIZE_HALF = 0.175  # Half position for take profit
    
    # Track position state
    position_side = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    tp_triggered = False  # Track if take profit already triggered
    
    for i in range(55, n):  # Start after EMA55 warmup
        # Get aligned daily trend
        ema21_d = ema21_aligned[i]
        ema55_d = ema55_aligned[i]
        
        # Get 4h indicators
        rsi_val = rsi[i]
        atr_val = atr[i]
        
        # Skip if any indicator is NaN
        if np.isnan(ema21_d) or np.isnan(ema55_d) or np.isnan(rsi_val) or np.isnan(atr_val):
            signals[i] = 0.0
            continue
        
        # Daily trend direction (EMA21 > EMA55 = bullish)
        daily_trend = 1 if ema21_d > ema55_d else -1
        
        # Entry logic: RSI pullback in trend direction (only when flat)
        if position_side == 0:
            # Long entry: daily bullish + RSI oversold pullback
            if daily_trend == 1 and rsi_val < 40:
                signals[i] = SIZE_FULL
                position_side = 1
                entry_price = close[i]
                highest_since_entry = close[i]
                tp_triggered = False
            
            # Short entry: daily bearish + RSI overbought pullback
            elif daily_trend == -1 and rsi_val > 60:
                signals[i] = -SIZE_FULL
                position_side = -1
                entry_price = close[i]
                lowest_since_entry = close[i]
                tp_triggered = False
        
        elif position_side == 1:  # Long position
            highest_since_entry = max(highest_since_entry, close[i])
            risk_multiple = 2.0  # 2ATR stoploss
            reward_multiple = 2.0  # 2R take profit
            
            # Calculate profit in ATR units
            profit_atr = (close[i] - entry_price) / atr_val if atr_val > 0 else 0
            
            # Take profit at 2R (reduce to half) - only trigger once
            if not tp_triggered and profit_atr >= reward_multiple * risk_multiple:
                signals[i] = SIZE_HALF
                tp_triggered = True
            
            # Trailing stoploss: exit if price drops from highest
            elif close[i] < highest_since_entry - 3 * atr_val:
                signals[i] = 0.0
                position_side = 0
            
            # Hard stoploss: exit if price moves 2ATR against entry
            elif close[i] < entry_price - risk_multiple * atr_val:
                signals[i] = 0.0
                position_side = 0
            
            else:
                # Maintain position (keep same signal to avoid churn)
                signals[i] = SIZE_HALF if tp_triggered else SIZE_FULL
        
        elif position_side == -1:  # Short position
            lowest_since_entry = min(lowest_since_entry, close[i])
            risk_multiple = 2.0
            reward_multiple = 2.0
            
            # Calculate profit in ATR units
            profit_atr = (entry_price - close[i]) / atr_val if atr_val > 0 else 0
            
            # Take profit at 2R (reduce to half) - only trigger once
            if not tp_triggered and profit_atr >= reward_multiple * risk_multiple:
                signals[i] = -SIZE_HALF
                tp_triggered = True
            
            # Trailing stoploss: exit if price rises from lowest
            elif close[i] > lowest_since_entry + 3 * atr_val:
                signals[i] = 0.0
                position_side = 0
            
            # Hard stoploss: exit if price moves 2ATR against entry
            elif close[i] > entry_price + risk_multiple * atr_val:
                signals[i] = 0.0
                position_side = 0
            
            else:
                # Maintain position (keep same signal to avoid churn)
                signals[i] = -SIZE_HALF if tp_triggered else -SIZE_FULL
    
    return signals