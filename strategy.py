#!/usr/bin/env python3
"""
EXPERIMENT #060 - HMA Trend + RSI Pullback + Weekly Filter (1d primary)
========================================================================
Hypothesis: Daily HMA trend following with RSI pullback entries captures
major crypto moves while avoiding chasing breakouts. Weekly HMA filter
ensures we trade with the major trend. This differs from Donchian breakouts
by entering on pullbacks (better risk/reward) rather than breakouts (often
false signals in chop).

Key features:
- Primary TF: 1d (daily bars = fewer but higher quality signals)
- HTF filter: 1w HMA(50) for major trend confirmation
- Trend: HMA(21/48) crossover + slope confirmation
- Entry: RSI(14) pullback to 40-55 zone in uptrend (or 45-60 in downtrend)
- Regime: Weekly HMA alignment + HMA slope > 0
- Stoploss: 2.5*ATR(14) trailing (wider for daily timeframe)
- Position sizing: 0.25 base, discrete levels (0.0, ±0.25, ±0.30)
- Take profit: Reduce to half at 2.5R profit, trail stop at 1.5R

Why this should beat Donchian (Sharpe=0.490):
- Pullback entries have better risk/reward than breakouts
- HMA is more responsive than Donchian channels
- Weekly filter prevents counter-trend trades in major reversals
- Conservative daily sizing controls drawdown in crypto volatility
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "hma_rsi_pullback_weekly_1d_1w_v2"
timeframe = "1d"
leverage = 1.0


def calculate_hma(close, period):
    """Calculate Hull Moving Average - more responsive than EMA"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, adjust=False, min_periods=period // 2).mean()
    wma2 = close_s.ewm(span=period, adjust=False, min_periods=period).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False, min_periods=int(np.sqrt(period))).mean()
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


def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing"""
    n = len(close)
    delta = np.diff(close, prepend=close[0])
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    rs = np.zeros(n)
    for i in range(n):
        if avg_loss[i] > 0:
            rs[i] = avg_gain[i] / avg_loss[i]
        else:
            rs[i] = 100
    
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_hma_slope(hma, lookback=5):
    """Calculate HMA slope (rate of change over lookback periods)"""
    n = len(hma)
    slope = np.zeros(n)
    for i in range(lookback, n):
        if hma[i - lookback] != 0:
            slope[i] = (hma[i] - hma[i - lookback]) / hma[i - lookback] * 100
    return slope


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 50)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    hma_fast = calculate_hma(close, 21)
    hma_slow = calculate_hma(close, 48)
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    
    # Calculate HMA slopes for trend confirmation
    hma_fast_slope = calculate_hma_slope(hma_fast, 5)
    hma_slow_slope = calculate_hma_slope(hma_slow, 5)
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Base position size (25% of capital)
    MAX_SIZE = 0.30   # Max position size with strong trend
    MIN_SIZE = 0.20   # Min position size
    HALF_SIZE = BASE_SIZE / 2
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    entry_atr = 0.0
    profit_target_hit = False
    trailing_stop_active = False
    
    min_period = 100  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_1w_aligned[i]) or np.isnan(hma_fast[i]) or 
            np.isnan(hma_slow[i]) or np.isnan(atr[i]) or np.isnan(rsi[i]) or
            np.isnan(hma_fast_slope[i]) or np.isnan(hma_slow_slope[i]) or
            atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # Weekly trend filter (major trend direction)
        price_above_1w_hma = close[i] > hma_1w_aligned[i]
        weekly_trend = 1 if price_above_1w_hma else -1
        
        # Daily trend confirmation
        hma_bullish = hma_fast[i] > hma_slow[i]
        hma_bearish = hma_fast[i] < hma_slow[i]
        
        # HMA slope confirmation (trend strength)
        fast_slope_positive = hma_fast_slope[i] > 0.5  # >0.5% per 5 days
        fast_slope_negative = hma_fast_slope[i] < -0.5
        slow_slope_positive = hma_slow_slope[i] > 0.2
        slow_slope_negative = hma_slow_slope[i] < -0.2
        
        # RSI pullback zones (not overbought/oversold, just resting)
        rsi_pullback_long = 40 <= rsi[i] <= 55  # Pullback in uptrend
        rsi_pullback_short = 45 <= rsi[i] <= 60  # Pullback in downtrend
        
        # Calculate position size based on trend strength
        trend_strength = 1.0
        if weekly_trend == 1 and fast_slope_positive and slow_slope_positive:
            trend_strength = 1.15
        elif weekly_trend == -1 and fast_slope_negative and slow_slope_negative:
            trend_strength = 1.15
        
        position_size = min(MAX_SIZE, max(MIN_SIZE, BASE_SIZE * trend_strength))
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: HMA bullish + Weekly bullish + RSI pullback + Positive slopes
        if (hma_bullish and weekly_trend == 1 and 
            rsi_pullback_long and fast_slope_positive and slow_slope_positive):
            target_signal = position_size
        
        # Short entry: HMA bearish + Weekly bearish + RSI pullback + Negative slopes
        elif (hma_bearish and weekly_trend == -1 and 
              rsi_pullback_short and fast_slope_negative and slow_slope_negative):
            target_signal = -position_size
        
        # Stoploss and take profit logic - check BEFORE setting new signal
        stoploss_triggered = False
        take_profit_triggered = False
        
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                
                # Trailing stop: 2.5*ATR from highest (wider for daily)
                trailing_stop = highest_since_entry - 2.5 * atr[i]
                
                # Check stoploss
                if close[i] < trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit (2.5R from entry, where R = 2.5*ATR at entry)
                if not profit_target_hit:
                    if close[i] >= entry_price + 6.25 * entry_atr:  # 2.5R = 6.25*ATR
                        take_profit_triggered = True
                
                # Activate trailing stop after 1.5R profit
                if not trailing_stop_active:
                    if close[i] >= entry_price + 3.75 * entry_atr:  # 1.5R
                        trailing_stop_active = True
                        
            else:
                # Short position - update lowest
                lowest_since_entry = min(lowest_since_entry, close[i])
                
                # Trailing stop: 2.5*ATR from lowest
                trailing_stop = lowest_since_entry + 2.5 * atr[i]
                
                # Check stoploss
                if close[i] > trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit
                if not profit_target_hit:
                    if close[i] <= entry_price - 6.25 * entry_atr:  # 2.5R profit
                        take_profit_triggered = True
                
                # Activate trailing stop after 1.5R profit
                if not trailing_stop_active:
                    if close[i] <= entry_price - 3.75 * entry_atr:  # 1.5R
                        trailing_stop_active = True
        
        if stoploss_triggered:
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            entry_price = 0.0
            entry_atr = 0.0
            profit_target_hit = False
            trailing_stop_active = False
            
        elif take_profit_triggered:
            # Reduce position to half at 2.5R profit
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
                trailing_stop_active = False
                
            elif position_side != 0:
                # Maintain existing position - check for trend reversal exit
                hma_reversal_long = hma_bearish  # Fast crossed below slow
                hma_reversal_short = hma_bullish  # Fast crossed above slow
                weekly_reversal = (position_side == 1 and weekly_trend == -1) or \
                                  (position_side == -1 and weekly_trend == 1)
                
                if hma_reversal_long or hma_reversal_short or weekly_reversal:
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    entry_price = 0.0
                    entry_atr = 0.0
                    profit_target_hit = False
                    trailing_stop_active = False
                else:
                    # Maintain position
                    if profit_target_hit:
                        signals[i] = HALF_SIZE * position_side
                    else:
                        signals[i] = position_size * position_side
            else:
                signals[i] = 0.0
    
    return signals