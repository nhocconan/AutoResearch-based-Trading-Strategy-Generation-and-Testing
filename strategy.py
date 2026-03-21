#!/usr/bin/env python3
"""
EXPERIMENT #055 - Supertrend + RSI Pullback + Volume Confirmation (15m primary)
=====================================================================================
Hypothesis: 15m supertrend captures intraday trends, but needs HTF filter to avoid chop.
4h HMA provides major trend direction. 1h RSI pullback (40-60 range) ensures we enter on
dips in uptrend rather than chasing breakouts. Volume confirmation (taker buy ratio > 0.55)
filters false signals. This differs from #047 by using supertrend (not Donchian) + RSI
pullback entries (not breakout entries) on faster 15m timeframe.

Key features:
- Primary TF: 15m
- HTF filters: 4h HMA(21) for trend, 1h RSI(14) for pullback detection
- Trend: Supertrend(ATR=10, mult=3) for entry timing
- Entry: Supertrend flip + RSI pullback (40-60 in uptrend, 40-60 in downtrend)
- Volume: Taker buy volume ratio > 0.55 for longs, < 0.45 for shorts
- Regime: 4h HMA slope filter (only trade with HTF trend)
- Stoploss: 2.0*ATR(14) trailing
- Position sizing: 0.25-0.30 discrete, scaled by RSI distance from 50

Why this should beat current best (Sharpe=0.490):
- 15m captures more intraday moves than 12h
- RSI pullback entries have better risk/reward than breakout entries
- Volume confirmation reduces false signals
- Conservative sizing (0.25-0.30) controls drawdown
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "supertrend_rsi_volume_15m_1h_4h_v1"
timeframe = "15m"
leverage = 1.0


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, adjust=False).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False).mean()
    return hma.values


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


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    hl2 = (high + low) / 2
    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr
    
    final_upper = np.zeros(n)
    final_lower = np.zeros(n)
    supertrend = np.zeros(n)
    trend = np.ones(n)  # 1 = bullish, -1 = bearish
    
    final_upper[0] = basic_upper[0]
    final_lower[0] = basic_lower[0]
    supertrend[0] = basic_upper[0]
    
    for i in range(1, n):
        # Calculate final upper/lower
        if basic_upper[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]:
            final_upper[i] = basic_upper[i]
        else:
            final_upper[i] = final_upper[i - 1]
        
        if basic_lower[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]:
            final_lower[i] = basic_lower[i]
        else:
            final_lower[i] = final_lower[i - 1]
        
        # Determine trend and supertrend value
        if trend[i - 1] == 1:
            if close[i] < final_lower[i]:
                trend[i] = -1
                supertrend[i] = final_upper[i]
            else:
                trend[i] = 1
                supertrend[i] = final_lower[i]
        else:
            if close[i] > final_upper[i]:
                trend[i] = 1
                supertrend[i] = final_lower[i]
            else:
                trend[i] = -1
                supertrend[i] = final_upper[i]
    
    return supertrend, trend


def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing"""
    n = len(close)
    delta = np.zeros(n)
    for i in range(1, n):
        delta[i] = close[i] - close[i - 1]
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    rsi = np.zeros(n)
    for i in range(period, n):
        if avg_loss[i] == 0:
            rsi[i] = 100
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_volume_ratio(taker_buy_volume, volume):
    """Calculate taker buy volume ratio"""
    ratio = np.zeros(len(volume))
    for i in range(len(volume)):
        if volume[i] > 0:
            ratio[i] = taker_buy_volume[i] / volume[i]
        else:
            ratio[i] = 0.5
    return ratio


def calculate_hma_slope(hma_values, lookback=5):
    """Calculate HMA slope (positive = uptrend, negative = downtrend)"""
    n = len(hma_values)
    slope = np.zeros(n)
    for i in range(lookback, n):
        if hma_values[i - lookback] != 0:
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback]
    return slope


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    taker_buy_volume = prices["taker_buy_volume"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1h = get_htf_data(prices, '1h')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    rsi_1h = calculate_rsi(df_1h['close'].values, 14)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h)
    
    # Calculate 4h HMA slope for regime filter
    hma_4h_slope = calculate_hma_slope(hma_4h_aligned, 5)
    
    # Calculate 15m indicators
    supertrend, st_trend = calculate_supertrend(high, low, close, 10, 3.0)
    atr = calculate_atr(high, low, close, 14)
    rsi_15m = calculate_rsi(close, 14)
    volume_ratio = calculate_volume_ratio(taker_buy_volume, volume)
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # Base position size (28% of capital)
    MAX_SIZE = 0.35   # Max position size
    MIN_SIZE = 0.20   # Min position size
    HALF_SIZE = BASE_SIZE / 2
    
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
        if (np.isnan(hma_4h_aligned[i]) or np.isnan(rsi_1h_aligned[i]) or
            np.isnan(supertrend[i]) or np.isnan(atr[i]) or np.isnan(rsi_15m[i]) or
            np.isnan(volume_ratio[i]) or np.isnan(hma_4h_slope[i]) or
            atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # 4h HMA trend filter
        price_above_4h_hma = close[i] > hma_4h_aligned[i]
        hma_4h_bullish = hma_4h_slope[i] > 0
        hma_4h_bearish = hma_4h_slope[i] < 0
        
        # 1h RSI pullback detection (40-60 range = neutral/pullback zone)
        rsi_1h_pullback_long = 40 <= rsi_1h_aligned[i] <= 60
        rsi_1h_pullback_short = 40 <= rsi_1h_aligned[i] <= 60
        
        # 15m Supertrend signals
        st_bullish = st_trend[i] == 1
        st_bearish = st_trend[i] == -1
        
        # Check for supertrend flip (entry trigger)
        st_flip_long = st_trend[i] == 1 and st_trend[i - 1] == -1
        st_flip_short = st_trend[i] == -1 and st_trend[i - 1] == 1
        
        # Volume confirmation
        volume_bullish = volume_ratio[i] > 0.55
        volume_bearish = volume_ratio[i] < 0.45
        
        # 15m RSI confirmation (not overbought/oversold at entry)
        rsi_15m_ok_long = rsi_15m[i] < 70
        rsi_15m_ok_short = rsi_15m[i] > 30
        
        # Calculate position size based on RSI distance from 50 (stronger signal = larger size)
        rsi_1h_dist = abs(rsi_1h_aligned[i] - 50)
        rsi_multiplier = 1.0 + (rsi_1h_dist / 50) * 0.25  # Max 1.25x
        position_size = min(MAX_SIZE, max(MIN_SIZE, BASE_SIZE * rsi_multiplier))
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: Supertrend flip + 4h bullish + 1h RSI pullback + volume confirmation
        if (st_flip_long and price_above_4h_hma and hma_4h_bullish and
            rsi_1h_pullback_long and volume_bullish and rsi_15m_ok_long):
            target_signal = position_size
        
        # Short entry: Supertrend flip + 4h bearish + 1h RSI pullback + volume confirmation
        elif (st_flip_short and not price_above_4h_hma and hma_4h_bearish and
              rsi_1h_pullback_short and volume_bearish and rsi_15m_ok_short):
            target_signal = -position_size
        
        # Stoploss and take profit logic - check BEFORE setting new signal
        stoploss_triggered = False
        take_profit_triggered = False
        
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.0 * atr[i]
                
                # Check stoploss
                if close[i] < trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit (2R from entry, where R = 2*ATR at entry)
                if not profit_target_hit:
                    if close[i] >= entry_price + 4.0 * entry_atr:  # 2R = 4*ATR
                        take_profit_triggered = True
            else:
                # Short position - update lowest
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + 2.0 * atr[i]
                
                # Check stoploss
                if close[i] > trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit
                if not profit_target_hit:
                    if close[i] <= entry_price - 4.0 * entry_atr:  # 2R profit
                        take_profit_triggered = True
        
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
                # Maintain existing position (check if trend reversed)
                # Exit if Supertrend reverses OR 4h trend breaks
                st_reversal_long = st_trend[i] == -1
                st_reversal_short = st_trend[i] == 1
                hma_trend_broken = (position_side == 1 and not price_above_4h_hma) or \
                                   (position_side == -1 and price_above_4h_hma)
                
                if st_reversal_long or st_reversal_short or hma_trend_broken:
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    entry_price = 0.0
                    entry_atr = 0.0
                    profit_target_hit = False
                else:
                    # Maintain position
                    signals[i] = position_size * position_side if not profit_target_hit else HALF_SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals