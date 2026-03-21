#!/usr/bin/env python3
"""
Experiment #302: 30m Regime-Adaptive Strategy with 4h Trend Bias + RSI + Volume
Hypothesis: 30m timeframe captures intraday swings while 4h HMA provides trend direction.
Using WIDER RSI ranges (25-75) ensures >=10 trades (learned from 0-trade failures).
Volume confirmation filters false breakouts. ATR stops at 2.5*ATR control drawdown.
Position size 0.25 balances returns vs risk (learned from -90% DD failures).
Target: Beat Sharpe=0.499 while ensuring >=10 trades per symbol on 30m timeframe.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_regime_4h_hma_rsi_volume_atr_v1"
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

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs moving average."""
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    return vol_ratio

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands for regime detection."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def calculate_choppiness_index(high, low, close, period=14):
    """Calculate Choppiness Index for regime detection."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    hl_range = high - low
    hl_sum = pd.Series(hl_range).rolling(window=period, min_periods=period).sum().values
    
    chop = 100 * np.log10(atr_sum / (hl_sum + 1e-10)) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_50 = calculate_hma(df_4h['close'].values, 50)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_50_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_50)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    rsi = calculate_rsi(close, 14)
    vol_ratio = calculate_volume_ratio(volume, 20)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    chop = calculate_choppiness_index(high, low, close, 14)
    
    # Track previous values for crossover detection
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    prev_hma_21 = np.roll(hma_21, 1)
    prev_hma_21[0] = hma_21[0]
    prev_rsi = np.roll(rsi, 1)
    prev_rsi[0] = rsi[0]
    prev_hma_4h = np.roll(hma_4h_21_aligned, 1)
    prev_hma_4h[0] = hma_4h_21_aligned[0]
    
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
        if np.isnan(hma_21[i]) or np.isnan(hma_50[i]) or np.isnan(atr[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # 4h macro trend bias
        trend_4h_bullish = close[i] > hma_4h_21_aligned[i] and hma_4h_21_aligned[i] > hma_4h_50_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_21_aligned[i] and hma_4h_21_aligned[i] < hma_4h_50_aligned[i]
        trend_4h_neutral = not trend_4h_bullish and not trend_4h_bearish
        
        # 30m trend filter
        trend_30m_bullish = close[i] > hma_21[i] and hma_21[i] > hma_50[i]
        trend_30m_bearish = close[i] < hma_21[i] and hma_21[i] < hma_50[i]
        
        # HMA slope
        hma_slope_bullish = hma_21[i] > prev_hma_21[i]
        hma_slope_bearish = hma_21[i] < prev_hma_21[i]
        
        # Regime detection via Choppiness Index
        is_ranging = chop[i] > 50  # CHOP > 50 = ranging market
        is_trending = chop[i] < 45  # CHOP < 45 = trending market
        
        # RSI zones (WIDER ranges to ensure trades)
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_neutral = 35 <= rsi[i] <= 65
        rsi_not_extreme_long = rsi[i] < 75
        rsi_not_extreme_short = rsi[i] > 25
        
        # Volume confirmation
        volume_confirmed = vol_ratio[i] > 1.0
        
        # Bollinger position
        near_bb_lower = close[i] < bb_lower[i] * 1.01
        near_bb_upper = close[i] > bb_upper[i] * 0.99
        bb_squeeze = (bb_upper[i] - bb_lower[i]) / bb_mid[i] < 0.05
        
        # Price crossover signals
        hma_cross_long = prev_close[i] <= prev_hma_21[i] and close[i] > hma_21[i]
        hma_cross_short = prev_close[i] >= prev_hma_21[i] and close[i] < hma_21[i]
        
        # 4h HMA crossover
        hma_4h_cross_long = prev_hma_4h[i] <= hma_4h_21_aligned[i] and close[i] > hma_4h_21_aligned[i]
        hma_4h_cross_short = prev_hma_4h[i] >= hma_4h_21_aligned[i] and close[i] < hma_4h_21_aligned[i]
        
        new_signal = 0.0
        
        # === LONG ENTRY (multiple conditions to ensure trades) ===
        # Primary: 4h bullish + 30m bullish + RSI pullback + volume
        if trend_4h_bullish and trend_30m_bullish and rsi_neutral and volume_confirmed:
            new_signal = SIZE_ENTRY
        # Secondary: 4h bullish + RSI oversold + price > HMA21 (pullback entry)
        elif trend_4h_bullish and rsi_oversold and close[i] > hma_21[i]:
            new_signal = SIZE_ENTRY
        # Tertiary: 30m bullish + HMA cross + volume (momentum entry)
        elif trend_30m_bullish and hma_cross_long and volume_confirmed:
            new_signal = SIZE_ENTRY
        # Quaternary: Ranging market + RSI oversold + near BB lower (mean reversion)
        elif is_ranging and rsi_oversold and near_bb_lower:
            new_signal = SIZE_ENTRY
        # Quinternary: 4h bullish + HMA slope up + RSI 40-60 (trend continuation)
        elif trend_4h_bullish and hma_slope_bullish and 40 < rsi[i] < 60:
            new_signal = SIZE_ENTRY
        # Simple fallback: 4h bullish + price > HMA21 + RSI > 40 (ensure trades)
        elif trend_4h_bullish and close[i] > hma_21[i] and rsi[i] > 40 and rsi_not_extreme_long:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRY (multiple conditions to ensure trades) ===
        # Primary: 4h bearish + 30m bearish + RSI pullback + volume
        if trend_4h_bearish and trend_30m_bearish and rsi_neutral and volume_confirmed:
            new_signal = -SIZE_ENTRY
        # Secondary: 4h bearish + RSI overbought + price < HMA21 (pullback entry)
        elif trend_4h_bearish and rsi_overbought and close[i] < hma_21[i]:
            new_signal = -SIZE_ENTRY
        # Tertiary: 30m bearish + HMA cross + volume (momentum entry)
        elif trend_30m_bearish and hma_cross_short and volume_confirmed:
            new_signal = -SIZE_ENTRY
        # Quaternary: Ranging market + RSI overbought + near BB upper (mean reversion)
        elif is_ranging and rsi_overbought and near_bb_upper:
            new_signal = -SIZE_ENTRY
        # Quinternary: 4h bearish + HMA slope down + RSI 40-60 (trend continuation)
        elif trend_4h_bearish and hma_slope_bearish and 40 < rsi[i] < 60:
            new_signal = -SIZE_ENTRY
        # Simple fallback: 4h bearish + price < HMA21 + RSI < 60 (ensure trades)
        elif trend_4h_bearish and close[i] < hma_21[i] and rsi[i] < 60 and rsi_not_extreme_short:
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