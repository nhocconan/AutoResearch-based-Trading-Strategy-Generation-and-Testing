#!/usr/bin/env python3
"""
Experiment #467: 12h ADX Regime + Daily HMA Bias + Dual Mode Entries
Hypothesis: Market has distinct regimes (trending vs ranging). ADX detects regime.
In trending regime (ADX>25): follow trend with RSI pullback entries.
In ranging regime (ADX<25): mean revert at Bollinger bands.
This dual-mode approach should generate MORE trades than pure trend following
while avoiding whipsaws in choppy markets. 12h TF reduces noise vs lower TFs.
Timeframe: 12h (REQUIRED), HTF: 1d via mtf_data helper.
Position sizing: 0.25 entry, 0.125 half (discrete levels for fee efficiency).
Stoploss: 2.5*ATR trailing stop to protect capital in 2022-style crashes.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_adx_regime_daily_hma_dual_mode_atr_v1"
timeframe = "12h"
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
    """Calculate Hull Moving Average for smoother trend with less lag."""
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
    """Calculate ADX for trend strength detection."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    up_move = high_s - high_s.shift(1)
    down_move = low_s.shift(1) - low_s
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean() / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values, plus_di.values, minus_di.values

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper.values, lower.values, sma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    hma_12h = calculate_hma(close, 21)
    rsi = calculate_rsi(close, 14)
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, 20, 2.0)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.125
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_12h[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            continue
        
        # Daily trend bias (HTF)
        daily_bullish = close[i] > hma_1d_aligned[i]
        daily_bearish = close[i] < hma_1d_aligned[i]
        
        # 12h HMA trend
        hma_12h_bullish = close[i] > hma_12h[i]
        hma_12h_bearish = close[i] < hma_12h[i]
        
        # ADX regime detection
        trending = adx[i] > 25
        ranging = adx[i] <= 25
        
        # DI crossover for trend direction
        di_bullish = plus_di[i] > minus_di[i]
        di_bearish = plus_di[i] < minus_di[i]
        
        # RSI zones
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_neutral_long = rsi[i] > 40 and rsi[i] < 55
        rsi_neutral_short = rsi[i] > 45 and rsi[i] < 60
        
        # Bollinger position
        near_bb_lower = close[i] < bb_lower[i] * 1.005  # within 0.5% of lower band
        near_bb_upper = close[i] > bb_upper[i] * 0.995  # within 0.5% of upper band
        below_bb_mid = close[i] < bb_mid[i]
        above_bb_mid = close[i] > bb_mid[i]
        
        new_signal = 0.0
        
        # === TRENDING REGIME (ADX > 25) ===
        if trending:
            # Long: Daily bullish + 12h bullish + DI bullish + RSI pullback
            if daily_bullish and hma_12h_bullish and di_bullish and rsi_neutral_long:
                new_signal = SIZE_ENTRY
            # Long: Daily bullish + 12h bullish + RSI oversold (deep pullback)
            elif daily_bullish and hma_12h_bullish and rsi_oversold:
                new_signal = SIZE_ENTRY
            # Long: Price above both HMA + DI bullish + RSI > 40
            elif close[i] > hma_12h[i] and close[i] > hma_1d_aligned[i] and di_bullish and rsi[i] > 40:
                new_signal = SIZE_ENTRY
            
            # Short: Daily bearish + 12h bearish + DI bearish + RSI pullback
            if daily_bearish and hma_12h_bearish and di_bearish and rsi_neutral_short:
                new_signal = -SIZE_ENTRY
            # Short: Daily bearish + 12h bearish + RSI overbought (rally short)
            elif daily_bearish and hma_12h_bearish and rsi_overbought:
                new_signal = -SIZE_ENTRY
            # Short: Price below both HMA + DI bearish + RSI < 60
            elif close[i] < hma_12h[i] and close[i] < hma_1d_aligned[i] and di_bearish and rsi[i] < 60:
                new_signal = -SIZE_ENTRY
        
        # === RANGING REGIME (ADX <= 25) ===
        else:
            # Long: Mean reversion at Bollinger lower + Daily bullish bias
            if near_bb_lower and daily_bullish:
                new_signal = SIZE_ENTRY
            # Long: Mean reversion at Bollinger lower + RSI oversold
            elif near_bb_lower and rsi_oversold:
                new_signal = SIZE_ENTRY
            # Long: Price near BB lower + below mid + RSI < 45
            elif near_bb_lower and below_bb_mid and rsi[i] < 45:
                new_signal = SIZE_ENTRY
            
            # Short: Mean reversion at Bollinger upper + Daily bearish bias
            if near_bb_upper and daily_bearish:
                new_signal = -SIZE_ENTRY
            # Short: Mean reversion at Bollinger upper + RSI overbought
            elif near_bb_upper and rsi_overbought:
                new_signal = -SIZE_ENTRY
            # Short: Price near BB upper + above mid + RSI > 55
            elif near_bb_upper and above_bb_mid and rsi[i] > 55:
                new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for 12h timeframe)
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
            
            # Calculate trailing stop (2.5*ATR for 12h timeframe)
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
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
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