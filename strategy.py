#!/usr/bin/env python3
"""
Experiment #045: 1h Multi-Signal Strategy with 4h/12h HMA Trend + Supertrend + RSI Pullback
Hypothesis: 1h timeframe balances trade frequency and noise. Using 4h HMA for intermediate
trend and 12h HMA for macro regime. Supertrend provides clear entry/exit signals while
RSI pullback entries (RSI<40 in uptrend, RSI>60 in downtrend) improve win rate.
ATR stoploss at 2.5x protects capital. Position sizing 0.25 with discrete levels minimizes
fee churn while generating sufficient trades (target 30-50/year).
Key improvement over #041: More entry triggers, relaxed RSI thresholds, better stoploss tracking.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_supertrend_4h_12h_hma_rsi_pullback_v1"
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
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
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

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator."""
    atr = calculate_atr(high, low, close, period)
    hl2 = (high + low) / 2
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr
    
    supertrend = np.zeros(len(close))
    direction = np.ones(len(close))
    
    supertrend[0] = lower[0]
    direction[0] = 1
    for i in range(1, len(close)):
        if close[i] > supertrend[i-1]:
            supertrend[i] = lower[i]
            direction[i] = 1
        elif close[i] < supertrend[i-1]:
            supertrend[i] = upper[i]
            direction[i] = -1
        else:
            supertrend[i] = supertrend[i-1]
            direction[i] = direction[i-1]
    
    return supertrend, direction

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    return close_s.rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_12h = calculate_hma(df_12h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3.0)
    sma_200 = calculate_sma(close, 200)
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    
    signals = np.zeros(n)
    SIZE = 0.25
    HALF_SIZE = 0.12
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    max_profit_r = 0.0
    
    for i in range(100, n):
        # Skip if any HTF data is NaN
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            continue
        
        # 12h macro regime (bull/bear)
        macro_bullish = close[i] > hma_12h_aligned[i]
        macro_bearish = close[i] < hma_12h_aligned[i]
        
        # 4h intermediate trend
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # 1h Supertrend direction
        st_long = st_direction[i] == 1
        st_short = st_direction[i] == -1
        
        # Supertrend flip signals (strongest entry)
        st_flip_long = i > 0 and st_direction[i] == 1 and st_direction[i-1] == -1
        st_flip_short = i > 0 and st_direction[i] == -1 and st_direction[i-1] == 1
        
        # 1h HMA trend
        hma_trend_long = hma_21[i] > hma_50[i]
        hma_trend_short = hma_21[i] < hma_50[i]
        
        # RSI signals (relaxed for more trades)
        rsi_oversold = rsi[i] < 40
        rsi_overbought = rsi[i] > 60
        rsi_rising = i > 2 and rsi[i] > rsi[i-2]
        rsi_falling = i > 2 and rsi[i] < rsi[i-2]
        
        # Price vs SMA200
        price_above_sma200 = close[i] > sma_200[i] if not np.isnan(sma_200[i]) else False
        price_below_sma200 = close[i] < sma_200[i] if not np.isnan(sma_200[i]) else False
        
        new_signal = 0.0
        
        # LONG ENTRY TRIGGERS (multiple paths)
        # Trigger 1: Supertrend flip long (strongest)
        if st_flip_long:
            new_signal = SIZE
        # Trigger 2: Macro bullish + 4h bullish + Supertrend long + RSI rising
        elif macro_bullish and trend_4h_bullish and st_long and rsi_rising:
            new_signal = SIZE
        # Trigger 3: RSI pullback in uptrend (RSI<40 + Supertrend long + macro bullish)
        elif rsi_oversold and st_long and macro_bullish:
            new_signal = SIZE
        # Trigger 4: HMA trend long + Supertrend long + price above SMA200
        elif hma_trend_long and st_long and price_above_sma200:
            new_signal = SIZE
        # Trigger 5: All three trends aligned (12h + 4h + 1h Supertrend)
        elif macro_bullish and trend_4h_bullish and st_long:
            new_signal = SIZE
        # Trigger 6: RSI rising from oversold + Supertrend long
        elif rsi_oversold and rsi_rising and st_long:
            new_signal = SIZE
        
        # SHORT ENTRY TRIGGERS
        # Trigger 1: Supertrend flip short (strongest)
        if st_flip_short:
            new_signal = -SIZE
        # Trigger 2: Macro bearish + 4h bearish + Supertrend short + RSI falling
        elif macro_bearish and trend_4h_bearish and st_short and rsi_falling:
            new_signal = -SIZE
        # Trigger 3: RSI pullback in downtrend (RSI>60 + Supertrend short + macro bearish)
        elif rsi_overbought and st_short and macro_bearish:
            new_signal = -SIZE
        # Trigger 4: HMA trend short + Supertrend short + price below SMA200
        elif hma_trend_short and st_short and price_below_sma200:
            new_signal = -SIZE
        # Trigger 5: All three trends aligned (12h + 4h + 1h Supertrend)
        elif macro_bearish and trend_4h_bearish and st_short:
            new_signal = -SIZE
        # Trigger 6: RSI falling from overbought + Supertrend short
        elif rsi_overbought and rsi_falling and st_short:
            new_signal = -SIZE
        
        # Stoploss and take profit logic (Rule 6)
        if position_side > 0 and entry_price > 0:
            stop_loss = entry_price - 2.5 * atr[i]
            if close[i] < stop_loss:
                new_signal = 0.0
            else:
                new_trailing = close[i] - 2.5 * atr[i]
                if new_trailing > trailing_stop:
                    trailing_stop = new_trailing
                if close[i] < trailing_stop and trailing_stop > entry_price:
                    new_signal = 0.0
                if close[i] > entry_price:
                    profit_r = (close[i] - entry_price) / atr[i]
                    if profit_r > max_profit_r:
                        max_profit_r = profit_r
                if max_profit_r >= 2.5 and signals[i-1] == SIZE:
                    new_signal = HALF_SIZE
        
        if position_side < 0 and entry_price > 0:
            stop_loss = entry_price + 2.5 * atr[i]
            if close[i] > stop_loss:
                new_signal = 0.0
            else:
                new_trailing = close[i] + 2.5 * atr[i]
                if new_trailing < trailing_stop or trailing_stop == 0:
                    trailing_stop = new_trailing
                if close[i] > trailing_stop and trailing_stop < entry_price:
                    new_signal = 0.0
                if close[i] < entry_price:
                    profit_r = (entry_price - close[i]) / atr[i]
                    if profit_r > max_profit_r:
                        max_profit_r = profit_r
                if max_profit_r >= 2.5 and signals[i-1] == -SIZE:
                    new_signal = -HALF_SIZE
        
        # Update position tracking
        if new_signal != 0 and position_side == 0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            max_profit_r = 0.0
        elif new_signal != 0 and position_side != 0:
            if np.sign(new_signal) != position_side:
                entry_price = close[i]
                position_side = np.sign(new_signal)
                trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
                max_profit_r = 0.0
        elif new_signal == 0 and position_side != 0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            max_profit_r = 0.0
        
        signals[i] = new_signal
    
    return signals