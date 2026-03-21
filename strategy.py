#!/usr/bin/env python3
"""
Experiment #277: 15m Multi-Timeframe Trend Following with 4h HMA Filter
Hypothesis: 15m strategies fail due to excessive whipsaws and fee churn. 
Solution: Use 4h HMA as PRIMARY trend filter (not optional), only trade in 4h trend direction.
1h RSI for pullback entry timing (loose thresholds: 35-65 range). 
ATR stoploss at 2.5x to survive volatility. Discrete signals (0, ±0.25, ±0.30) to minimize churn.
Target: 30-50 trades/year, Sharpe > 0.5, DD < -30%
Key insight from failures: Mean reversion dies on 15m. Pure trend following with strong HTF filter works.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_trend_4h_hma_1h_rsi_atr_v1"
timeframe = "15m"
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
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
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

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_1h = calculate_hma(df_1h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    sma_50 = calculate_sma(close, 50)
    
    # Track previous values
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    prev_rsi = np.roll(rsi, 1)
    prev_rsi[0] = rsi[0]
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.30
    SIZE_HALF = 0.15
    SIZE_EXIT = 0.0
    
    # Track positions for stoploss/takeprofit
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # === PRIMARY TREND FILTER (4h HMA) ===
        # This is the KEY - only trade in 4h trend direction
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # === SECONDARY FILTER (1h HMA) ===
        trend_1h_bullish = close[i] > hma_1h_aligned[i]
        trend_1h_bearish = close[i] < hma_1h_aligned[i]
        
        # === LOCAL TREND (15m SMA50) ===
        local_bullish = close[i] > sma_50[i] if not np.isnan(sma_50[i]) else False
        local_bearish = close[i] < sma_50[i] if not np.isnan(sma_50[i]) else False
        
        # === RSI PULLBACK SIGNALS (loose thresholds for more trades) ===
        # Long: RSI pulled back from overbought but still in bullish zone
        rsi_pullback_long = (35 < rsi[i] < 55) and (prev_rsi[i] <= 40 or prev_rsi[i] < rsi[i])
        rsi_oversold_bounce = rsi[i] < 35 and prev_rsi[i] <= 30
        
        # Short: RSI pulled back from oversold but still in bearish zone
        rsi_pullback_short = (45 < rsi[i] < 65) and (prev_rsi[i] >= 60 or prev_rsi[i] > rsi[i])
        rsi_overbought_reject = rsi[i] > 65 and prev_rsi[i] >= 70
        
        # === MOMENTUM CONFIRMATION ===
        momentum_up = close[i] > prev_close[i]
        momentum_down = close[i] < prev_close[i]
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Must have 4h bullish trend (PRIMARY filter)
        if trend_4h_bullish:
            # Strong entry: 4h + 1h + local all bullish + RSI pullback
            if trend_1h_bullish and local_bullish:
                if rsi_pullback_long or rsi_oversold_bounce:
                    if momentum_up:
                        new_signal = SIZE_ENTRY
            # Moderate entry: 4h bullish + RSI very oversold
            elif rsi_oversold_bounce and momentum_up:
                new_signal = SIZE_ENTRY
            # Continuation: all trends aligned + RSI neutral-bullish
            elif trend_1h_bullish and rsi[i] > 45 and rsi[i] < 60:
                if momentum_up and close[i] > sma_50[i]:
                    new_signal = SIZE_ENTRY
        
        # === SHORT ENTRY ===
        # Must have 4h bearish trend (PRIMARY filter)
        if trend_4h_bearish:
            # Strong entry: 4h + 1h + local all bearish + RSI pullback
            if trend_1h_bearish and local_bearish:
                if rsi_pullback_short or rsi_overbought_reject:
                    if momentum_down:
                        new_signal = -SIZE_ENTRY
            # Moderate entry: 4h bearish + RSI very overbought
            elif rsi_overbought_reject and momentum_down:
                new_signal = -SIZE_ENTRY
            # Continuation: all trends aligned + RSI neutral-bearish
            elif trend_1h_bearish and rsi[i] < 55 and rsi[i] > 40:
                if momentum_down and close[i] < sma_50[i]:
                    new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing stop
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from highest)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = SIZE_EXIT
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing stop
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from lowest)
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = SIZE_EXIT
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = entry_price - close[i]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reduced (take profit)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals