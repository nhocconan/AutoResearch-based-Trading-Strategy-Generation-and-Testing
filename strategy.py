#!/usr/bin/env python3
"""
Experiment #063: 1h RSI Pullback with 4h HMA Trend Filter + ADX Confirmation
Hypothesis: Simpler is better after 56 failed experiments. Use 4h HMA(21) for 
trend direction (faster response than daily), 1h RSI(14) pullback for entry 
timing in established trends, ADX(14) to confirm trend strength (>15). 
Avoid over-filtering that caused failures in #057 (Sharpe=-2.9), #062 (Sharpe=-3.6).
Key insight: Current best uses Supertrend+Daily HMA+RSI. We simplify by using
4h HMA instead of daily (faster signals) and ADX instead of Supertrend (cleaner).
Position sizing: 0.25-0.30 discrete levels, 2.0*ATR stoploss, no complex TP logic.
Target: 40-80 trades/year per symbol, Sharpe > 0.5, DD < -35%.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_rsi_pullback_4h_hma_adx_v1"
timeframe = "1h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    half = period // 2
    if half < 1:
        half = 1
    sqrt_period = int(np.sqrt(period))
    if sqrt_period < 1:
        sqrt_period = 1
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate RSI indicator."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_adx(high, low, close, period=14):
    """Calculate ADX indicator for trend strength."""
    plus_dm = np.zeros(len(close))
    minus_dm = np.zeros(len(close))
    
    for i in range(1, len(close)):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        elif minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    atr = calculate_atr(high, low, close, period)
    atr_safe = np.where(atr > 0, atr, 1e-10)
    
    plus_di = 100 * pd.Series(plus_dm / atr_safe).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_di = 100 * pd.Series(minus_dm / atr_safe).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    di_sum = plus_di + minus_di
    di_sum_safe = np.where(di_sum > 0, di_sum, 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / di_sum_safe
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1 - critical for performance)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h HMA for trend direction
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars only)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    adx = calculate_adx(high, low, close, 14)
    
    # 1h HMA for additional trend confirmation
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.30
    SIZE_HALF = 0.15
    
    # Track position state for stoploss management
    position_side = 0
    entry_price = 0.0
    stop_price = 0.0
    position_reduced = False
    
    for i in range(100, n):
        # 4h trend filter (HTF bias)
        hma_4h_valid = hma_4h_aligned[i] > 0
        trend_bullish = hma_4h_valid and close[i] > hma_4h_aligned[i]
        trend_bearish = hma_4h_valid and close[i] < hma_4h_aligned[i]
        
        # ADX trend strength filter (avoid choppy markets)
        trend_strong = adx[i] > 15
        
        # 1h HMA trend confirmation
        hma_trend_long = hma_21[i] > hma_50[i]
        hma_trend_short = hma_21[i] < hma_50[i]
        
        # RSI pullback signals (wider ranges to ensure sufficient trades)
        # Long: RSI dipped from overbought but not oversold
        rsi_pullback_long = rsi[i] >= 30 and rsi[i] <= 50
        # Short: RSI rallied from oversold but not overbought
        rsi_pullback_short = rsi[i] >= 50 and rsi[i] <= 70
        
        # RSI momentum confirmation
        rsi_rising = (i > 2) and (rsi[i] > rsi[i-2])
        rsi_falling = (i > 2) and (rsi[i] < rsi[i-2])
        
        new_signal = 0.0
        
        # LONG ENTRY: 4h bullish + ADX strong + RSI pullback + 1h HMA confirmation
        if trend_bullish and trend_strong and rsi_pullback_long and hma_trend_long:
            new_signal = SIZE_ENTRY
        # Alternative long: 4h bullish + RSI rising from deep pullback
        elif trend_bullish and rsi[i] < 40 and rsi_rising:
            new_signal = SIZE_ENTRY
        
        # SHORT ENTRY: 4h bearish + ADX strong + RSI pullback + 1h HMA confirmation
        if trend_bearish and trend_strong and rsi_pullback_short and hma_trend_short:
            new_signal = -SIZE_ENTRY
        # Alternative short: 4h bearish + RSI falling from high levels
        elif trend_bearish and rsi[i] > 60 and rsi_falling:
            new_signal = -SIZE_ENTRY
        
        # Stoploss logic (Rule 6) - MUST check before position tracking update
        if position_side > 0 and entry_price > 0:
            # Update trailing stop (only move up for longs)
            current_stop = close[i] - 2.0 * atr[i]
            if current_stop > stop_price:
                stop_price = current_stop
            
            # Check if stoploss hit
            if close[i] < stop_price:
                new_signal = 0.0
            # Check take profit (reduce at 2R)
            elif not position_reduced and new_signal != 0.0:
                profit = close[i] - entry_price
                risk = 2.0 * atr[i]
                if risk > 0 and profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update trailing stop (only move down for shorts)
            current_stop = close[i] + 2.0 * atr[i]
            if stop_price == 0.0 or current_stop < stop_price:
                stop_price = current_stop
            
            # Check if stoploss hit
            if close[i] > stop_price:
                new_signal = 0.0
            # Check take profit (reduce at 2R)
            elif not position_reduced and new_signal != 0.0:
                profit = entry_price - close[i]
                risk = 2.0 * atr[i]
                if risk > 0 and profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened (was flat, now has signal)
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            if position_side > 0:
                stop_price = close[i] - 2.0 * atr[i]
            else:
                stop_price = close[i] + 2.0 * atr[i]
            position_reduced = False
        
        # Position reversed (switched from long to short or vice versa)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            if position_side > 0:
                stop_price = close[i] - 2.0 * atr[i]
            else:
                stop_price = close[i] + 2.0 * atr[i]
            position_reduced = False
        
        # Position reduced (take profit partial exit)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        # Position closed (signal went to zero)
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            stop_price = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals