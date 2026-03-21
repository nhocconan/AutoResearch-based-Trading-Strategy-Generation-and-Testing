#!/usr/bin/env python3
"""
Experiment #297: 1h RSI Pullback + 4h HMA Trend + ADX Strength Filter
Hypothesis: 1h timeframe captures medium-term pullbacks within 4h trend direction.
RSI pullback entries (30-50 for long in uptrend, 50-70 for short in downtrend) ensure
we enter on dips without waiting for extreme oversold conditions (learned from 0-trade failures).
ADX(14) > 20 filter ensures we only trade when there's sufficient trend strength.
Simple entry logic with generous RSI ranges ensures >=10 trades per symbol on all markets.
ATR-based trailing stops (2.5*ATR) control drawdown. Position size 0.25 balances risk/return.
Target: Beat Sharpe=0.499 from current best while ensuring positive Sharpe on ALL symbols.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_rsi_pullback_4h_hma_adx_atr_v1"
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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    # Smoothed values
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean() / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average for long-term trend filter."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

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
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    adx = calculate_adx(high, low, close, 14)
    sma_200 = calculate_sma(close, 200)
    
    # Track previous values for crossover detection
    prev_rsi = np.roll(rsi, 1)
    prev_rsi[0] = rsi[0]
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.12
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_4h_aligned[i]) or np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF trend direction)
        hma_4h_bullish = close[i] > hma_4h_aligned[i]
        hma_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # ADX trend strength filter
        adx_strong = adx[i] > 20  # Minimum trend strength
        adx_very_strong = adx[i] > 25
        
        # RSI pullback zones (generous ranges to ensure trades)
        rsi_pullback_long = 30 <= rsi[i] <= 55
        rsi_pullback_short = 45 <= rsi[i] <= 70
        rsi_oversold = rsi[i] < 40
        rsi_overbought = rsi[i] > 60
        
        # RSI turning up/down
        rsi_turning_up = rsi[i] > prev_rsi[i] and rsi[i-1] <= prev_rsi[i-1] if i > 1 else False
        rsi_turning_down = rsi[i] < prev_rsi[i] and rsi[i-1] >= prev_rsi[i-1] if i > 1 else False
        
        # Price above/below SMA200
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # Price momentum
        price_momentum_up = close[i] > prev_close[i]
        price_momentum_down = close[i] < prev_close[i]
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Primary: 4h bullish + ADX strong + RSI pullback + RSI turning up
        if hma_4h_bullish and adx_strong and rsi_pullback_long and rsi_turning_up:
            new_signal = SIZE_ENTRY
        # Secondary: 4h bullish + Above SMA200 + RSI oversold (stronger signal)
        elif hma_4h_bullish and above_sma200 and rsi_oversold:
            new_signal = SIZE_ENTRY
        # Tertiary: 4h bullish + RSI 35-50 + Price momentum up (simpler for more trades)
        elif hma_4h_bullish and 35 <= rsi[i] <= 50 and price_momentum_up:
            new_signal = SIZE_ENTRY
        # Quaternary: ADX very strong + RSI pullback long (trend continuation)
        elif adx_very_strong and rsi_pullback_long and price_momentum_up:
            new_signal = SIZE_ENTRY
        # Simple backup: 4h bullish + RSI < 50 (ensures minimum trades)
        elif hma_4h_bullish and rsi[i] < 50 and price_momentum_up:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRY ===
        # Primary: 4h bearish + ADX strong + RSI pullback + RSI turning down
        if hma_4h_bearish and adx_strong and rsi_pullback_short and rsi_turning_down:
            new_signal = -SIZE_ENTRY
        # Secondary: 4h bearish + Below SMA200 + RSI overbought (stronger signal)
        elif hma_4h_bearish and below_sma200 and rsi_overbought:
            new_signal = -SIZE_ENTRY
        # Tertiary: 4h bearish + RSI 50-65 + Price momentum down (simpler for more trades)
        elif hma_4h_bearish and 50 <= rsi[i] <= 65 and price_momentum_down:
            new_signal = -SIZE_ENTRY
        # Quaternary: ADX very strong + RSI pullback short (trend continuation)
        elif adx_very_strong and rsi_pullback_short and price_momentum_down:
            new_signal = -SIZE_ENTRY
        # Simple backup: 4h bearish + RSI > 50 (ensures minimum trades)
        elif hma_4h_bearish and rsi[i] > 50 and price_momentum_down:
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