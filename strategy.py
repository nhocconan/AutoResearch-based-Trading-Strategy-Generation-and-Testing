#!/usr/bin/env python3
"""
Experiment #307: 15m Multi-Timeframe Mean Reversion with Trend Filter
Hypothesis: 15m is too noisy for pure trend-following. Instead, use 4h HMA for macro bias,
1h RSI for pullback extremes, and 15m EMA crossover for precise entry timing.
ADX filter avoids dead markets. This combines trend direction (HTF) with mean reversion (LTF).
Position size 0.28 with 2.5*ATR stops. Target: Beat Sharpe=0.499 with >=10 trades/symbol.
Key insight from failures: shorter TFs need stronger HTF filters + generous entry conditions.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_4h_hma_1h_rsi_ema_crossover_atr_v1"
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

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    ema = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    return ema

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    plus_dm = np.zeros(len(close))
    minus_dm = np.zeros(len(close))
    
    for i in range(1, len(close)):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        elif minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / np.maximum(atr, 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / np.maximum(atr, 1e-10)
    
    dx = 100 * np.abs(plus_di - minus_di) / np.maximum(plus_di + minus_di, 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate 4h HMA for macro trend
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h RSI for pullback detection
    rsi_1h = calculate_rsi(df_1h['close'].values, 14)
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h)
    
    # Calculate 1h ADX for trend strength
    adx_1h, _, _ = calculate_adx(df_1h['high'].values, df_1h['low'].values, df_1h['close'].values, 14)
    adx_1h_aligned = align_htf_to_ltf(prices, df_1h, adx_1h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    ema_8 = calculate_ema(close, 8)
    ema_21 = calculate_ema(close, 21)
    rsi_15m = calculate_rsi(close, 14)
    
    # Previous values for crossover detection
    prev_ema_8 = np.roll(ema_8, 1)
    prev_ema_21 = np.roll(ema_21, 1)
    prev_ema_8[0] = ema_8[0]
    prev_ema_21[0] = ema_21[0]
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.28
    SIZE_HALF = 0.14
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(ema_8[i]) or np.isnan(ema_21[i]):
            signals[i] = 0.0
            continue
        
        # 4h macro trend bias
        hma_4h_valid = not np.isnan(hma_4h_aligned[i])
        trend_bullish = hma_4h_valid and close[i] > hma_4h_aligned[i]
        trend_bearish = hma_4h_valid and close[i] < hma_4h_aligned[i]
        
        # 1h RSI pullback zones (generous for more trades)
        rsi_1h_valid = not np.isnan(rsi_1h_aligned[i])
        rsi_oversold = rsi_1h_valid and rsi_1h_aligned[i] < 45
        rsi_overbought = rsi_1h_valid and rsi_1h_aligned[i] > 55
        
        # 1h ADX trend strength (avoid dead markets)
        adx_valid = not np.isnan(adx_1h_aligned[i])
        trending_market = adx_valid and adx_1h_aligned[i] > 18
        
        # 15m EMA crossover signals
        ema_bullish_cross = ema_8[i] > ema_21[i] and prev_ema_8[i] <= prev_ema_21[i]
        ema_bearish_cross = ema_8[i] < ema_21[i] and prev_ema_8[i] >= prev_21[i]
        
        # 15m EMA alignment (already in trend)
        ema_aligned_long = ema_8[i] > ema_21[i]
        ema_aligned_short = ema_8[i] < ema_21[i]
        
        # 15m RSI confirmation
        rsi_15m_long = rsi_15m[i] < 60
        rsi_15m_short = rsi_15m[i] > 40
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Primary: 4h bullish + 1h RSI pullback + 15m EMA cross + ADX trending
        if trend_bullish and rsi_oversold and ema_bullish_cross and trending_market:
            new_signal = SIZE_ENTRY
        # Secondary: 4h bullish + 15m EMA aligned + 1h RSI not overbought
        elif trend_bullish and ema_aligned_long and rsi_1h_valid and rsi_1h_aligned[i] < 65 and rsi_15m_long:
            new_signal = SIZE_ENTRY
        # Tertiary: 4h bullish + EMA cross + RSI < 55 (simpler for more trades)
        elif trend_bullish and ema_bullish_cross and rsi_15m[i] < 55:
            new_signal = SIZE_ENTRY
        # Quaternary: Price > 4h HMA + EMA aligned + RSI 35-55 (trend continuation)
        elif trend_bullish and ema_aligned_long and 35 < rsi_15m[i] < 55:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRY ===
        # Primary: 4h bearish + 1h RSI pullback + 15m EMA cross + ADX trending
        if trend_bearish and rsi_overbought and ema_bearish_cross and trending_market:
            new_signal = -SIZE_ENTRY
        # Secondary: 4h bearish + 15m EMA aligned + 1h RSI not oversold
        elif trend_bearish and ema_aligned_short and rsi_1h_valid and rsi_1h_aligned[i] > 35 and rsi_15m_short:
            new_signal = -SIZE_ENTRY
        # Tertiary: 4h bearish + EMA cross + RSI > 45 (simpler for more trades)
        elif trend_bearish and ema_bearish_cross and rsi_15m[i] > 45:
            new_signal = -SIZE_ENTRY
        # Quaternary: Price < 4h HMA + EMA aligned + RSI 45-65 (trend continuation)
        elif trend_bearish and ema_aligned_short and 45 < rsi_15m[i] < 65:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from highest)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from lowest)
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
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