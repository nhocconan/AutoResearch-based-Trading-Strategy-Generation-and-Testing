#!/usr/bin/env python3
"""
EXPERIMENT #002 - HMA Trend + KAMA Momentum + Volume Filter (30m primary, 4h HTF)
=================================================================================
Hypothesis: 30m HMA(21) captures intermediate trend direction with less lag than EMA.
4h HMA(50) provides higher timeframe trend alignment to avoid counter-trend trades.
KAMA(14) adaptive momentum confirms trend strength (KAMA rises in trends, flattens in chop).
Volume filter ensures we only enter on above-average volume bars (avoids false breakouts).
ATR(14) trailing stop at 2.5x protects capital during reversals.

Key differences from failed supertrend_rsi strategy:
- HMA/KAMA combo instead of Supertrend (smoother, less whipsaw)
- Volume confirmation filter (reduces false entries)
- More selective entries = fewer trades, less fee churn
- Discrete position sizing: 0.0, ±0.25, ±0.30 only

Position sizing: 0.25 base, 0.30 strong conviction
Stoploss: 2.5*ATR(14) trailing
Take profit: Reduce to half at 2R, trail stop at 1R
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "hma_kama_volume_30m_4h_v1"
timeframe = "30m"
leverage = 1.0


def calculate_hma(close, period):
    """Calculate Hull Moving Average - reduces lag vs EMA"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, adjust=False).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False).mean()
    return hma.values


def calculate_kama(close, high, low, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average
    Adapts smoothing based on market efficiency (trend vs chop)
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(close - np.roll(close, er_period))
    change[:er_period] = np.nan
    
    volatility = np.zeros(n)
    for i in range(er_period, n):
        volatility[i] = np.sum(np.abs(np.diff(close[i-er_period:i+1])))
    
    er = change / (volatility + 1e-10)
    er[:er_period] = np.nan
    
    # Calculate smoothing constant
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    # Calculate KAMA iteratively
    for i in range(er_period + 1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
    atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    return atr


def calculate_volume_sma(volume, period=20):
    """Calculate volume simple moving average"""
    vol_s = pd.Series(volume)
    vol_sma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_sma


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    hma_4h_raw = calculate_hma(df_4h['close'].values, 50)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)  # auto shift(1)
    
    # Calculate 30m indicators (primary timeframe)
    hma_30m = calculate_hma(close, 21)
    kama_30m = calculate_kama(close, high, low, er_period=10)
    atr = calculate_atr(high, low, close, 14)
    vol_sma = calculate_volume_sma(volume, 20)
    
    # Generate signals
    signals = np.zeros(n)
    SIZE_BASE = 0.25  # Base position size (25% of capital)
    SIZE_STRONG = 0.30  # Strong conviction size
    HALF_SIZE = SIZE_BASE / 2
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    profit_target_hit = False
    entry_atr = 0.0
    
    min_period = 100  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_4h_aligned[i]) or np.isnan(hma_30m[i]) or 
            np.isnan(kama_30m[i]) or np.isnan(atr[i]) or np.isnan(vol_sma[i]) or 
            atr[i] == 0 or vol_sma[i] == 0):
            signals[i] = 0.0
            continue
        
        # 4h HTF trend filter (major trend direction)
        htf_trend = 1 if close[i] > hma_4h_aligned[i] else -1
        
        # 30m HMA trend (intermediate trend)
        ltf_trend = 1 if close[i] > hma_30m[i] else -1
        
        # KAMA momentum confirmation (KAMA slope)
        kama_slope = kama_30m[i] - kama_30m[i-5] if i >= 5 else 0
        kama_bullish = kama_slope > 0
        kama_bearish = kama_slope < 0
        
        # Volume filter (only trade on above-average volume)
        volume_ratio = volume[i] / vol_sma[i] if vol_sma[i] > 0 else 0
        volume_confirmed = volume_ratio > 1.0  # Above average volume
        
        # Determine target signal based on all filters
        target_signal = 0.0
        conviction = 1.0  # Default conviction level
        
        # Long entry: HTF bullish + LTF bullish + KAMA rising + Volume confirmed
        if htf_trend == 1 and ltf_trend == 1 and kama_bullish and volume_confirmed:
            # Strong conviction if all aligned
            if kama_slope > atr[i] * 0.5:  # Strong KAMA momentum
                target_signal = SIZE_STRONG
            else:
                target_signal = SIZE_BASE
        
        # Short entry: HTF bearish + LTF bearish + KAMA falling + Volume confirmed
        elif htf_trend == -1 and ltf_trend == -1 and kama_bearish and volume_confirmed:
            # Strong conviction if all aligned
            if kama_slope < -atr[i] * 0.5:  # Strong KAMA momentum
                target_signal = -SIZE_STRONG
            else:
                target_signal = -SIZE_BASE
        
        # Stoploss and take profit logic - check BEFORE setting new signal
        stoploss_triggered = False
        take_profit_triggered = False
        trend_reversal = False
        
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.5 * entry_atr
                
                # Check stoploss
                if close[i] < trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit (2R from entry, where R = 2.5*ATR)
                if not profit_target_hit and entry_atr > 0:
                    if close[i] >= entry_price + 5.0 * entry_atr:  # 2R = 5*ATR
                        take_profit_triggered = True
                
                # Check trend reversal (HTF or LTF flipped bearish)
                if htf_trend == -1 or ltf_trend == -1:
                    trend_reversal = True
                    
            else:
                # Short position - update lowest
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + 2.5 * entry_atr
                
                # Check stoploss
                if close[i] > trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit
                if not profit_target_hit and entry_atr > 0:
                    if close[i] <= entry_price - 5.0 * entry_atr:  # 2R profit
                        take_profit_triggered = True
                
                # Check trend reversal (HTF or LTF flipped bullish)
                if htf_trend == 1 or ltf_trend == 1:
                    trend_reversal = True
        
        # Apply signal based on conditions
        if stoploss_triggered:
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            entry_price = 0.0
            entry_atr = 0.0
            profit_target_hit = False
            
        elif take_profit_triggered:
            # Reduce position to half at 2R profit
            signals[i] = HALF_SIZE * position_side
            profit_target_hit = True
            
        elif trend_reversal:
            # Trend reversed, exit position
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            entry_price = 0.0
            entry_atr = 0.0
            profit_target_hit = False
            
        else:
            # Apply signal change
            if target_signal != 0.0 and position_side == 0:
                # New entry
                signals[i] = target_signal
                position_side = 1 if target_signal > 0 else -1
                highest_since_entry = close[i]
                lowest_since_entry = close[i]
                entry_price = close[i]
                entry_atr = atr[i]
                profit_target_hit = False
            elif position_side != 0:
                # Maintain existing position
                if profit_target_hit:
                    signals[i] = HALF_SIZE * position_side
                else:
                    signals[i] = SIZE_BASE * position_side if abs(target_signal) == SIZE_BASE else SIZE_STRONG * position_side
            else:
                signals[i] = 0.0
    
    return signals