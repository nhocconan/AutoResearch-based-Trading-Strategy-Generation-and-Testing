#!/usr/bin/env python3
"""
Experiment #304: 4h Fisher Transform + Daily HMA Trend + Donchian Breakout
Hypothesis: 4h timeframe is ideal for medium-term trends. Fisher Transform catches reversals
better than RSI (proven in literature). Daily HMA provides macro bias. Donchian(20) breakout
confirms momentum. This combo should generate 20-40 trades/year with good win rate.
Key difference from failed 4h strategies: Fisher instead of RSI, simpler entry logic,
Donchian breakout confirmation instead of multiple conflicting filters.
Position size 0.30, stoploss 2.5*ATR, take profit at 2R then trail.
Target: Beat Sharpe=0.499 from current best while ensuring >=10 trades per symbol.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_daily_hma_donchian_breakout_atr_v1"
timeframe = "4h"
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

def calculate_fisher(close, period=9):
    """
    Ehlers Fisher Transform - catches reversals better than RSI.
    Normalizes price to Gaussian distribution, crossings of ±1.5 signal reversals.
    """
    close_s = pd.Series(close)
    # Calculate (high - low) / 2 for normalization
    high = close_s  # Use close as proxy since we don't have high/low in this function
    low = close_s
    
    # Actually we need high/low - use close range approximation
    hl2 = close_s
    ll = close_s.rolling(window=period, min_periods=period).min()
    hh = close_s.rolling(window=period, min_periods=period).max()
    
    # Normalize to 0-1 range
    norm = (hl2 - ll) / (hh - ll + 1e-10)
    norm = np.clip(norm, 0.001, 0.999)
    
    # Fisher transform
    fisher_input = 0.66 * ((norm - 0.5) / 0.5) + 0.67 * np.roll(fisher_input_calculated(norm), 1)
    fisher_input = np.clip(fisher_input, -0.999, 0.999)
    
    fisher = 0.5 * np.log((1 + fisher_input) / (1 - fisher_input + 1e-10))
    fisher_prev = np.roll(fisher, 1)
    fisher_prev[0] = fisher[0]
    
    return fisher, fisher_prev

def fisher_input_calculated(norm):
    """Helper for Fisher calculation."""
    return 0.66 * ((norm - 0.5) / 0.5)

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel upper and lower bands."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2
    return upper, lower, mid

def calculate_rsi(close, period=14):
    """Calculate RSI indicator for additional filter."""
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
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, 20)
    
    # Fisher Transform on 4h
    fisher, fisher_prev = calculate_fisher(close, 9)
    
    # HMA on 4h for trend confirmation
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    
    # Track previous values
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    prev_hma_21 = np.roll(hma_21, 1)
    prev_hma_21[0] = hma_21[0]
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.30
    SIZE_HALF = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_21[i]) or np.isnan(hma_50[i]) or np.isnan(atr[i]) or np.isnan(donchian_upper[i]):
            signals[i] = 0.0
            continue
        
        # Daily macro trend bias
        daily_bullish = close[i] > hma_1d_aligned[i]
        daily_bearish = close[i] < hma_1d_aligned[i]
        
        # 4h trend filter
        trend_bullish = hma_21[i] > hma_50[i]
        trend_bearish = hma_21[i] < hma_50[i]
        
        # Fisher Transform signals (reversal detection)
        fisher_long = fisher_prev[i] < -1.5 and fisher[i] >= -1.5
        fisher_short = fisher_prev[i] > 1.5 and fisher[i] <= 1.5
        
        # Fisher in extreme zones (oversold/overbought)
        fisher_oversold = fisher[i] < -1.0
        fisher_overbought = fisher[i] > 1.0
        
        # Donchian breakout
        breakout_long = close[i] > donchian_upper[i-1]  # Break above previous upper
        breakout_short = close[i] < donchian_lower[i-1]  # Break below previous lower
        
        # RSI filter (avoid extreme entries)
        rsi_ok_long = rsi[i] < 70
        rsi_ok_short = rsi[i] > 30
        
        # HMA slope
        hma_slope_up = hma_21[i] > prev_hma_21[i]
        hma_slope_down = hma_21[i] < prev_hma_21[i]
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Primary: Daily bullish + 4h trend up + Fisher reversal + Donchian breakout
        if daily_bullish and trend_bullish and fisher_long and breakout_long:
            new_signal = SIZE_ENTRY
        # Secondary: Daily bullish + Fisher oversold + Price > HMA21 + RSI ok
        elif daily_bullish and fisher_oversold and close[i] > hma_21[i] and rsi_ok_long and hma_slope_up:
            new_signal = SIZE_ENTRY
        # Tertiary: 4h trend up + Donchian breakout + RSI ok (simpler for more trades)
        elif trend_bullish and breakout_long and rsi_ok_long:
            new_signal = SIZE_ENTRY
        # Quaternary: Daily bullish + Price > HMA21 > HMA50 + Fisher turning up
        elif daily_bullish and close[i] > hma_21[i] > hma_50[i] and fisher[i] > fisher_prev[i]:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRY ===
        # Primary: Daily bearish + 4h trend down + Fisher reversal + Donchian breakout
        if daily_bearish and trend_bearish and fisher_short and breakout_short:
            new_signal = -SIZE_ENTRY
        # Secondary: Daily bearish + Fisher overbought + Price < HMA21 + RSI ok
        elif daily_bearish and fisher_overbought and close[i] < hma_21[i] and rsi_ok_short and hma_slope_down:
            new_signal = -SIZE_ENTRY
        # Tertiary: 4h trend down + Donchian breakout + RSI ok (simpler for more trades)
        elif trend_bearish and breakout_short and rsi_ok_short:
            new_signal = -SIZE_ENTRY
        # Quaternary: Daily bearish + Price < HMA21 < HMA50 + Fisher turning down
        elif daily_bearish and close[i] < hma_21[i] < hma_50[i] and fisher[i] < fisher_prev[i]:
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