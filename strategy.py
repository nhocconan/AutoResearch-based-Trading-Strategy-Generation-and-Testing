#!/usr/bin/env python3
"""
EXPERIMENT #027 - HTF Trend + RSI Pullback + ADX Filter (1h primary, 4h HTF)
================================================================================
Hypothesis: Combining lessons from best performer (dual_htf_trend_pullback) with
1h timeframe. 4h HMA provides major trend filter, RSI pullback entries reduce
chase risk, ADX ensures minimum trend strength. This differs from failed supertrend
attempts by using simpler HMA trend + RSI timing + ADX confirmation.

Key features:
- Primary TF: 1h (required for this experiment)
- HTF filter: 4h HMA(21) for trend direction (loaded ONCE before loop)
- Entry: RSI(14) pullback to 45-55 zone in trend direction
- Strength: ADX(14) > 22 (avoid weak trends)
- Regime: BB Width > 45th percentile (avoid chop)
- Stoploss: 2.5*ATR(14) trailing
- Position sizing: 0.25 base, discrete levels (0.0, ±0.25, ±0.35)
- Take profit: Half position at 2R, trail stop
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "htf_rsi_pullback_adx_1h_4h_v1"
timeframe = "1h"
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


def calculate_adx(high, low, close, period=14):
    """Calculate ADX"""
    n = len(close)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(high[i] - high[i-1], 0)
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(low[i-1] - low[i], 0)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
    
    atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values / (atr + 1e-10)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    return adx


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values


def calculate_bollinger_bands(close, period=20, std_dev=2):
    """Calculate Bollinger Bands and Band Width"""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    band_width = (upper - lower) / sma
    return upper.values, lower.values, band_width.values


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


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)  # auto shift(1)
    
    # Calculate 1h indicators (pre-compute before loop for performance)
    atr = calculate_atr(high, low, close, 14)
    adx = calculate_adx(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_width = calculate_bollinger_bands(close, 20, 2)
    bb_width_pr = calculate_percentile_rank(bb_width, 100)
    
    # Generate signals
    signals = np.zeros(n)
    SIZE = 0.25  # Base position size (25% of capital - conservative)
    HALF_SIZE = SIZE / 2  # For take profit reduction
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    profit_target_hit = False
    
    min_period = 150  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator (Rule 5 - no look-ahead)
        if (np.isnan(hma_4h_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(adx[i]) or np.isnan(rsi[i]) or np.isnan(bb_width_pr[i]) or 
            atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # 4h HTF trend filter (major trend direction)
        hma_trend = 1 if close[i] > hma_4h_aligned[i] else -1
        
        # ADX trend strength filter (avoid weak/choppy trends)
        adx_valid = adx[i] > 22
        
        # Regime filter: only trade when BB Width is in top 55% (trending market)
        regime_valid = bb_width_pr[i] > 0.55
        
        # RSI pullback zone (45-55 for entry timing - neutral zone pullback)
        rsi_neutral = 45 <= rsi[i] <= 55
        
        # RSI momentum confirmation
        rsi_momentum_bull = rsi[i] > rsi[i-1] if i > 0 else False
        rsi_momentum_bear = rsi[i] < rsi[i-1] if i > 0 else False
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: 4h trend bullish + ADX strong + RSI neutral + RSI rising + Regime valid
        if (hma_trend == 1 and adx_valid and rsi_neutral and 
            rsi_momentum_bull and regime_valid):
            target_signal = SIZE
        
        # Short entry: 4h trend bearish + ADX strong + RSI neutral + RSI falling + Regime valid
        elif (hma_trend == -1 and adx_valid and rsi_neutral and 
              rsi_momentum_bear and regime_valid):
            target_signal = -SIZE
        
        # Stoploss and take profit logic - check BEFORE setting new signal (Rule 6)
        stoploss_triggered = False
        take_profit_triggered = False
        
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.5 * atr[i]
                
                # Check stoploss
                if close[i] < trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit (2R from entry, where R = 2.5*ATR)
                if not profit_target_hit:
                    if close[i] >= entry_price + 5.0 * atr[i]:  # 2R = 5*ATR
                        take_profit_triggered = True
            else:
                # Short position - update lowest
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + 2.5 * atr[i]
                
                # Check stoploss
                if close[i] > trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit
                if not profit_target_hit:
                    if close[i] <= entry_price - 5.0 * atr[i]:  # 2R profit
                        take_profit_triggered = True
        
        if stoploss_triggered:
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            entry_price = 0.0
            profit_target_hit = False
        elif take_profit_triggered:
            # Reduce position to half at 2R profit (Rule 4 - discrete levels)
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
                profit_target_hit = False
            elif position_side != 0:
                # Maintain existing position (check if HTF trend reversed)
                if position_side == 1 and hma_trend == -1:
                    # 4h trend reversed, exit long
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    entry_price = 0.0
                    profit_target_hit = False
                elif position_side == -1 and hma_trend == 1:
                    # 4h trend reversed, exit short
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    entry_price = 0.0
                    profit_target_hit = False
                else:
                    # Maintain position at current size
                    signals[i] = SIZE * position_side if not profit_target_hit else HALF_SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals