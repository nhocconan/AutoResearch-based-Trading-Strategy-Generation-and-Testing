#!/usr/bin/env python3
"""
Experiment #080: 30m Trend-Following with 4h HMA Filter + RSI Momentum
Hypothesis: Previous 30m strategies failed due to too many conflicting filters.
Simplify to: 4h HMA trend direction + 30m EMA state + RSI momentum confirmation.
Fewer filters = more trades. Use proven 4h HMA from best strategy (Sharpe=0.499).
Position sizing: 0.25 entry, 0.15 at 1.5R profit, stoploss at 2.5*ATR trailing.
Ensure entry conditions are loose enough to generate 10+ trades per symbol.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_ema_rsi_4h_hma_trend_v1"
timeframe = "30m"
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

def calculate_ema(close, period):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    
    # EMA for trend state
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_price = 0.0
    lowest_price = 0.0
    entry_atr = 0.0
    
    for i in range(100, n):
        # 4h trend filter (HTF) - price relative to 4h HMA
        hma_4h_val = hma_4h_aligned[i]
        if np.isnan(hma_4h_val) or hma_4h_val == 0:
            daily_bullish = True
            daily_bearish = False
        else:
            daily_bullish = close[i] > hma_4h_val
            daily_bearish = close[i] < hma_4h_val
        
        # 30m EMA trend state
        ema_trend_long = ema_21[i] > ema_50[i]
        ema_trend_short = ema_21[i] < ema_50[i]
        
        # EMA crossover signals (for entry timing)
        ema_cross_long = False
        ema_cross_short = False
        if i > 0:
            ema_cross_long = ema_21[i] > ema_50[i] and ema_21[i-1] <= ema_50[i-1]
            ema_cross_short = ema_21[i] < ema_50[i] and ema_21[i-1] >= ema_50[i-1]
        
        # RSI momentum (not extreme - just direction)
        rsi_momentum_long = rsi[i] > 45 and rsi[i] < 70
        rsi_momentum_short = rsi[i] < 55 and rsi[i] > 30
        
        # RSI crossover for entry timing
        rsi_cross_long = False
        rsi_cross_short = False
        if i > 0:
            rsi_cross_long = rsi[i] > 50 and rsi[i-1] <= 50
            rsi_cross_short = rsi[i] < 50 and rsi[i-1] >= 50
        
        new_signal = 0.0
        
        # LONG ENTRY - simpler conditions to ensure trades
        # Primary: EMA trend + 4h bullish + RSI momentum
        if ema_trend_long and daily_bullish and rsi_momentum_long:
            new_signal = SIZE_ENTRY
        # Secondary: EMA cross + 4h bullish (catch early trends)
        elif ema_cross_long and daily_bullish:
            new_signal = SIZE_ENTRY
        # Tertiary: RSI cross + EMA trend (momentum entry)
        elif rsi_cross_long and ema_trend_long:
            new_signal = SIZE_ENTRY
        
        # SHORT ENTRY - simpler conditions to ensure trades
        # Primary: EMA trend + 4h bearish + RSI momentum
        if ema_trend_short and daily_bearish and rsi_momentum_short:
            new_signal = -SIZE_ENTRY
        # Secondary: EMA cross + 4h bearish (catch early trends)
        elif ema_cross_short and daily_bearish:
            new_signal = -SIZE_ENTRY
        # Tertiary: RSI cross + EMA trend (momentum entry)
        elif rsi_cross_short and ema_trend_short:
            new_signal = -SIZE_ENTRY
        
        # Stoploss logic (Rule 6) - check BEFORE updating position tracking
        if position_side > 0 and entry_price > 0:
            # Update highest price for trailing
            if close[i] > highest_price:
                highest_price = close[i]
            
            # Calculate trailing stop (2.5*ATR from highest)
            current_stop = highest_price - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 1.5R
                risk = entry_atr * 2.5
                profit = close[i] - entry_price
                if profit >= 1.5 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest price for trailing
            if lowest_price == 0.0 or close[i] < lowest_price:
                lowest_price = close[i]
            
            # Calculate trailing stop (2.5*ATR from lowest)
            current_stop = lowest_price + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 1.5R
                risk = entry_atr * 2.5
                profit = entry_price - close[i]
                if profit >= 1.5 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            entry_atr = atr[i]
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_price = close[i] if position_side > 0 else 0.0
            lowest_price = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            entry_atr = atr[i]
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_price = close[i] if position_side > 0 else 0.0
            lowest_price = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reduced (take profit)
        elif new_signal != 0.0 and prev_signal != 0.0 and abs(new_signal) < abs(prev_signal):
            position_reduced = True
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            entry_atr = 0.0
            trailing_stop = 0.0
            highest_price = 0.0
            lowest_price = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals