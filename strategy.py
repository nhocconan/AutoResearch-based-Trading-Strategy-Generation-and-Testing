#!/usr/bin/env python3
"""
EXPERIMENT #053 - Keltner Channel Breakout + Volatility Regime + Triple HTF Filter (12h primary)
=====================================================================================
Hypothesis: Keltner Channels (EMA + ATR bands) provide smoother breakout signals than Donchian
(high/low extremes). Combined with volatility regime filter (avoid extreme ATR percentiles)
and triple HTF alignment (12h/1d/1w), this should reduce false breakouts while capturing
major trend moves. Key difference from #047: Keltner vs Donchian, volatility regime filter,
tighter stoploss (1.5*ATR), faster trend reversal exit.

Key features:
- Primary TF: 12h
- HTF filters: 1d EMA(50) + 1w EMA(50) for triple alignment
- Trend: Keltner Channel(EMA20, ATR14*2.0) breakout
- Regime: ATR percentile 20-80 (avoid extreme volatility)
- Entry: Close breaks Keltner upper/lower with HTF alignment
- Stoploss: 1.5*ATR(14) trailing (tighter than #047's 2.0*ATR)
- Position sizing: 0.25-0.30 discrete, scaled by ATR percentile
- Take profit: Reduce to half at 2R profit, trail stop at 1R

Why this should beat #047 (Sharpe=0.490):
- Keltner Channels smoother than Donchian = fewer whipsaws
- Volatility regime filter avoids trading during extreme moves
- Tighter stoploss (1.5*ATR) reduces drawdown on failed breakouts
- EMA slope confirmation adds momentum filter
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "keltner_volregime_triplehtf_12h_1d_1w_v1"
timeframe = "12h"
leverage = 1.0


def calculate_ema(close, period):
    """Calculate Exponential Moving Average"""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, adjust=False, min_periods=period).mean()
    return ema.values


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


def calculate_keltner(high, low, close, ema_period=20, atr_period=14, atr_mult=2.0):
    """Calculate Keltner Channel (EMA + ATR bands)"""
    ema = calculate_ema(close, ema_period)
    atr = calculate_atr(high, low, close, atr_period)
    
    upper = ema + (atr_mult * atr)
    lower = ema - (atr_mult * atr)
    
    return upper, lower, ema


def calculate_ema_slope(ema, lookback=5):
    """Calculate EMA slope (rate of change over lookback periods)"""
    n = len(ema)
    slope = np.zeros(n)
    slope[:] = np.nan
    
    for i in range(lookback, n):
        if ema[i - lookback] != 0:
            slope[i] = (ema[i] - ema[i - lookback]) / ema[i - lookback]
    
    return slope


def calculate_percentile_rank(series, window=100):
    """Calculate rolling percentile rank"""
    n = len(series)
    pr = np.zeros(n)
    pr[:] = np.nan
    
    for i in range(window - 1, n):
        if not np.isnan(series[i]):
            window_data = series[i - window + 1:i + 1]
            window_data = window_data[~np.isnan(window_data)]
            if len(window_data) > 0:
                pr[i] = np.sum(window_data <= series[i]) / len(window_data)
    
    return pr


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    n = len(close)
    delta = np.zeros(n)
    for i in range(1, n):
        delta[i] = close[i] - close[i - 1]
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = 50.0
    
    for i in range(period, n):
        if avg_loss[i] == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))
    
    return rsi


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    ema_1d = calculate_ema(df_1d['close'].values, 50)
    ema_1w = calculate_ema(df_1w['close'].values, 50)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate 12h indicators
    keltner_upper, keltner_lower, keltner_ema = calculate_keltner(high, low, close, 20, 14, 2.0)
    atr = calculate_atr(high, low, close, 14)
    ema_slope = calculate_ema_slope(keltner_ema, 5)
    rsi = calculate_rsi(close, 14)
    
    # Calculate ATR percentile (volatility regime filter)
    atr_pr = calculate_percentile_rank(atr, 100)
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Base position size (25% of capital)
    MAX_SIZE = 0.32   # Max position size with favorable volatility
    MIN_SIZE = 0.18   # Min position size
    HALF_SIZE = BASE_SIZE / 2
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    profit_target_hit = False
    entry_atr = 0.0
    
    min_period = 150  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(ema_1w_aligned[i]) or
            np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]) or
            np.isnan(keltner_ema[i]) or np.isnan(atr[i]) or np.isnan(atr_pr[i]) or
            np.isnan(ema_slope[i]) or np.isnan(rsi[i]) or
            atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # Triple HTF trend alignment
        price_above_1d_ema = close[i] > ema_1d_aligned[i]
        price_above_1w_ema = close[i] > ema_1w_aligned[i]
        
        # 1d and 1w trend direction
        daily_trend = 1 if price_above_1d_ema else -1
        weekly_trend = 1 if price_above_1w_ema else -1
        
        # Volatility regime filter (avoid extreme volatility - only trade in 20-80 percentile)
        vol_normal = 0.20 <= atr_pr[i] <= 0.80
        
        # Keltner breakout signals
        breakout_long = close[i] > keltner_upper[i - 1]  # Break above previous upper
        breakout_short = close[i] < keltner_lower[i - 1]  # Break below previous lower
        
        # EMA slope confirmation (momentum)
        ema_bullish = ema_slope[i] > 0.001  # Positive slope
        ema_bearish = ema_slope[i] < -0.001  # Negative slope
        
        # RSI filter (avoid extreme overbought/oversold on entry)
        rsi_ok_long = rsi[i] < 75  # Not extremely overbought
        rsi_ok_short = rsi[i] > 25  # Not extremely oversold
        
        # Calculate position size based on volatility regime (dynamic sizing)
        # Lower volatility = larger position, higher volatility = smaller position
        if atr_pr[i] < 0.50:
            vol_multiplier = 1.15  # Low vol = increase size
        elif atr_pr[i] > 0.70:
            vol_multiplier = 0.85  # High vol = decrease size
        else:
            vol_multiplier = 1.0
        
        position_size = min(MAX_SIZE, max(MIN_SIZE, BASE_SIZE * vol_multiplier))
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: Keltner breakout + Vol normal + EMA bullish + Triple HTF bullish + RSI ok
        if (breakout_long and vol_normal and ema_bullish and 
            daily_trend == 1 and weekly_trend == 1 and rsi_ok_long):
            target_signal = position_size
        
        # Short entry: Keltner breakout + Vol normal + EMA bearish + Triple HTF bearish + RSI ok
        elif (breakout_short and vol_normal and ema_bearish and 
              daily_trend == -1 and weekly_trend == -1 and rsi_ok_short):
            target_signal = -position_size
        
        # Stoploss and take profit logic - check BEFORE setting new signal
        stoploss_triggered = False
        take_profit_triggered = False
        
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 1.5 * atr[i]  # Tighter stop than #047
                
                # Check stoploss
                if close[i] < trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit (2R from entry, where R = 1.5*ATR at entry)
                if not profit_target_hit:
                    if close[i] >= entry_price + 3.0 * entry_atr:  # 2R = 3*ATR (with 1.5*ATR stop)
                        take_profit_triggered = True
            else:
                # Short position - update lowest
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + 1.5 * atr[i]
                
                # Check stoploss
                if close[i] > trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit
                if not profit_target_hit:
                    if close[i] <= entry_price - 3.0 * entry_atr:  # 2R profit
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
                # Exit if Keltner reverses OR HTF alignment breaks OR EMA slope reverses
                keltner_reversal_long = close[i] < keltner_ema[i]
                keltner_reversal_short = close[i] > keltner_ema[i]
                hma_alignment_broken = (position_side == 1 and daily_trend == -1) or \
                                       (position_side == -1 and daily_trend == 1)
                ema_slope_reversal = (position_side == 1 and ema_bearish) or \
                                     (position_side == -1 and ema_bullish)
                
                if keltner_reversal_long or keltner_reversal_short or hma_alignment_broken or ema_slope_reversal:
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