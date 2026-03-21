#!/usr/bin/env python3
"""
EXPERIMENT #002 - Multi-Timeframe Trend + Pullback Strategy
============================================================
Hypothesis: Daily EMA trend filter (21/55) provides cleaner directional bias than 4h alone.
4h RSI pullback entries capture mean-reversion within the daily trend, reducing whipsaws.
ATR-based stoploss protects capital during trend reversals.

Key features:
- 1d EMA(21/55) crossover for primary trend direction
- 4h RSI(14) < 45 for long entries, > 55 for short entries (pullback logic)
- ATR(14) trailing stoploss at 2.5x ATR
- Discrete position sizing: 0.0, ±0.30 to minimize fee churn
- Max position: 30% of capital (critical for drawdown control)

Why this should work:
- Daily trend filter eliminates 4h noise and false signals
- RSI pullback entries avoid chasing breakouts
- Higher TF = fewer trades = less fee impact (0.10% round trip)
- Stoploss prevents catastrophic drawdowns like 2022 crash
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_daily_trend_4h_rsi_v1"
timeframe = "4h"
leverage = 1.0


def calculate_ema(series: np.ndarray, span: int) -> np.ndarray:
    """Calculate EMA with proper min_periods"""
    return pd.Series(series).ewm(span=span, min_periods=span, adjust=False).mean().values


def calculate_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate RSI with proper min_periods"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean().values
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
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
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # =========================================================
    # LOAD HTF DATA ONCE BEFORE LOOP (Rule 1 - CRITICAL)
    # =========================================================
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily EMA trend indicators
    close_1d = df_1d['close'].values
    ema_21_1d = calculate_ema(close_1d, 21)
    ema_55_1d = calculate_ema(close_1d, 55)
    
    # Determine daily trend direction (1 = bullish, -1 = bearish, 0 = neutral)
    daily_trend_raw = np.zeros(len(close_1d))
    for i in range(55, len(close_1d)):
        if ema_21_1d[i] > ema_55_1d[i]:
            daily_trend_raw[i] = 1
        elif ema_21_1d[i] < ema_55_1d[i]:
            daily_trend_raw[i] = -1
    
    # Align daily trend to 4h timeframe (auto shift(1) for completed bars)
    daily_trend = align_htf_to_ltf(prices, df_1d, daily_trend_raw)
    
    # =========================================================
    # CALCULATE 4H INDICATORS (primary timeframe)
    # =========================================================
    rsi_4h = calculate_rsi(close, 14)
    atr_4h = calculate_atr(high, low, close, 14)
    
    # =========================================================
    # GENERATE SIGNALS WITH STOPLOSS TRACKING
    # =========================================================
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size - conservative for drawdown control
    STOP_MULT = 2.5  # 2.5x ATR stoploss
    
    # Track position state for stoploss
    position_side = 0  # 0 = flat, 1 = long, -1 = short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup period for all indicators
    warmup = max(55, 14)
    
    for i in range(warmup, n):
        # Skip if indicators are NaN
        if np.isnan(rsi_4h[i]) or np.isnan(atr_4h[i]) or np.isnan(daily_trend[i]):
            signals[i] = 0.0
            continue
        
        current_trend = daily_trend[i]
        current_rsi = rsi_4h[i]
        current_atr = atr_4h[i]
        current_close = close[i]
        
        # ============================================
        # STOPLOSS LOGIC (Rule 6 - CRITICAL)
        # ============================================
        if position_side == 1:  # Long position
            # Update highest price since entry
            highest_since_entry = max(highest_since_entry, current_close)
            
            # Trailing stoploss: exit if price drops 2.5x ATR from highest
            stop_price = highest_since_entry - (STOP_MULT * current_atr)
            
            if current_close < stop_price:
                signals[i] = 0.0
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                continue
            
            # Take profit: reduce to half at 2R profit (2 * 2.5 ATR = 5 ATR)
            profit_target = entry_price + (5.0 * current_atr)
            if current_close >= profit_target and signals[i-1] == SIZE:
                signals[i] = SIZE / 2  # Reduce to half position
                continue
        
        elif position_side == -1:  # Short position
            # Update lowest price since entry
            lowest_since_entry = min(lowest_since_entry, current_close)
            
            # Trailing stoploss: exit if price rises 2.5x ATR from lowest
            stop_price = lowest_since_entry + (STOP_MULT * current_atr)
            
            if current_close > stop_price:
                signals[i] = 0.0
                position_side = 0
                entry_price = 0.0
                lowest_since_entry = 0.0
                continue
            
            # Take profit: reduce to half at 2R profit
            profit_target = entry_price - (5.0 * current_atr)
            if current_close <= profit_target and signals[i-1] == -SIZE:
                signals[i] = -SIZE / 2  # Reduce to half position
                continue
        
        # ============================================
        # ENTRY LOGIC (only if flat)
        # ============================================
        if position_side == 0:
            # Long entry: Daily trend bullish + RSI pullback
            if current_trend == 1 and current_rsi < 45:
                signals[i] = SIZE
                position_side = 1
                entry_price = current_close
                highest_since_entry = current_close
            
            # Short entry: Daily trend bearish + RSI pullback
            elif current_trend == -1 and current_rsi > 55:
                signals[i] = -SIZE
                position_side = -1
                entry_price = current_close
                lowest_since_entry = current_close
        else:
            # Hold existing position (signal already set from stoploss/TP logic above)
            if position_side == 1 and signals[i] == 0:
                signals[i] = SIZE  # Maintain long
            elif position_side == -1 and signals[i] == 0:
                signals[i] = -SIZE  # Maintain short
    
    return signals