#!/usr/bin/env python3
"""
EXPERIMENT #009 - EMA Crossover + HTF Trend Filter + Volume Confirmation (1h)
==============================================================================
Hypothesis: Combining 1h EMA(8/21) crossover signals with 12h trend filter and 
volume confirmation will reduce false breakouts while capturing sustained trends.
The 12h EMA provides stronger trend filter than 4h, while volume confirmation
ensures we only trade breakouts with institutional participation.

Key features:
- 12h EMA(21) for trend direction (loaded ONCE before loop via mtf_data)
- 1h EMA(8)/EMA(21) crossover for entry timing
- Volume ratio filter (>1.2x average) to confirm breakouts
- ATR(14) trailing stoploss at 2*ATR
- Discrete position sizing: 0.0, ±0.25 (25% of capital)
- Take profit: reduce to half at 2R profit
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "ema_crossover_htf_volume_1h_v1"
timeframe = "1h"
leverage = 1.0


def calculate_ema(close, period):
    """Calculate Exponential Moving Average with proper min_periods"""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values


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


def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs rolling average"""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean()
    ratio = vol_s / (vol_avg + 1e-10)
    return ratio.values


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load 12h HTF data ONCE before loop (CRITICAL - Rule 1)
    df_12h = get_htf_data(prices, '12h')
    ema_12h = calculate_ema(df_12h['close'].values, 21)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 1h indicators
    ema_fast = calculate_ema(close, 8)
    ema_slow = calculate_ema(close, 21)
    atr = calculate_atr(high, low, close, 14)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size - conservative for drawdown control
    
    entry_price = 0.0
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    tp_triggered = False
    
    for i in range(50, n):
        # Check for NaN values
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(ema_fast[i]) or 
            np.isnan(ema_slow[i]) or np.isnan(atr[i]) or np.isnan(vol_ratio[i]) or 
            atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # 12h trend filter
        trend_12h = ema_12h_aligned[i]
        trend_bullish = close[i] > trend_12h
        trend_bearish = close[i] < trend_12h
        
        # EMA crossover signals
        ema_bullish_cross = ema_fast[i] > ema_slow[i] and ema_fast[i-1] <= ema_slow[i-1]
        ema_bearish_cross = ema_fast[i] < ema_slow[i] and ema_fast[i-1] >= ema_slow[i-1]
        
        # Volume confirmation (must be >1.2x average)
        volume_confirmed = vol_ratio[i] > 1.2
        
        if position_side == 0:
            # Enter long: bullish 12h trend + EMA bullish cross + volume confirmation
            if trend_bullish and ema_bullish_cross and volume_confirmed:
                signals[i] = SIZE
                entry_price = close[i]
                position_side = 1
                highest_close = close[i]
                tp_triggered = False
            # Enter short: bearish 12h trend + EMA bearish cross + volume confirmation
            elif trend_bearish and ema_bearish_cross and volume_confirmed:
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
            # Trend reversal: exit if price crosses below 12h EMA
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
            # Trend reversal: exit if price crosses above 12h EMA
            elif not trend_bearish:
                signals[i] = 0.0
                position_side = 0
                entry_price = 0.0
                tp_triggered = False
            else:
                signals[i] = -SIZE if not tp_triggered else -SIZE / 2
    
    return signals