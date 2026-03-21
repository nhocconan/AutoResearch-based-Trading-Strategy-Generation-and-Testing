#!/usr/bin/env python3
"""
Experiment #151: 15m Multi-Timeframe Regime-Adaptive Strategy with 4h/1h Filters
Hypothesis: 15m timeframe needs strong HTF filters to avoid noise whipsaws.
Using 4h HMA for major trend bias + 1h RSI for pullback timing + Bollinger Band
regime detection (squeeze vs expansion). Long when 4h bullish + 1h RSI oversold +
BB squeeze breaking out. Short when 4h bearish + 1h RSI overbought + BB expansion.
This combines trend-following (HTF) with mean-reversion (RSI pullbacks) for
better risk-adjusted returns in both bull and bear markets. ATR stoploss at 2.5x
protects capital. Position sizing: 0.25 entry, 0.125 at 2R profit (discrete levels).
Key innovation: Bollinger Band Width percentile detects regime - only enter breakouts
when BBW expanding from squeeze, avoid choppy markets.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_regime_4h_hma_1h_rsi_bbw_v1"
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

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bbw = np.where(sma > 0, (upper - lower) / sma * 100, 0.0)
    return upper, lower, bbw

def calculate_bb_percentile(bbw, lookback=100):
    """Calculate BBW percentile for regime detection."""
    bbw_s = pd.Series(bbw)
    bbw_percentile = bbw_s.rolling(window=lookback, min_periods=lookback).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5,
        raw=False
    ).values
    bbw_percentile = np.nan_to_num(bbw_percentile, nan=0.5)
    return bbw_percentile

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs recent average."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean().values
    vol_ratio = np.where(vol_avg > 0, volume / vol_avg, 1.0)
    return vol_ratio

def calculate_momentum(close, period=10):
    """Calculate Rate of Change momentum."""
    momentum = np.zeros(len(close))
    for i in range(period, len(close)):
        if close[i-period] > 0:
            momentum[i] = (close[i] - close[i-period]) / close[i-period] * 100
        else:
            momentum[i] = 0.0
    return momentum

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    rsi_1h = calculate_rsi(df_1h['close'].values, 14)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi_15m = calculate_rsi(close, 14)
    bb_upper, bb_lower, bbw = calculate_bollinger(close, 20, 2.0)
    bbw_pct = calculate_bb_percentile(bbw, 100)
    vol_ratio = calculate_volume_ratio(volume, 20)
    momentum = calculate_momentum(close, 10)
    hma_20 = calculate_hma(close, 20)
    hma_50 = calculate_hma(close, 50)
    
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
        # 4h trend filter (major trend direction)
        hma_4h_valid = hma_4h_aligned[i] > 0
        trend_4h_bullish = hma_4h_valid and close[i] > hma_4h_aligned[i]
        trend_4h_bearish = hma_4h_valid and close[i] < hma_4h_aligned[i]
        
        # 1h RSI for pullback timing
        rsi_1h_oversold = rsi_1h_aligned[i] < 45
        rsi_1h_overbought = rsi_1h_aligned[i] > 55
        rsi_1h_neutral = 40 < rsi_1h_aligned[i] < 60
        
        # 15m RSI for entry timing
        rsi_15m_oversold = rsi_15m[i] < 35
        rsi_15m_overbought = rsi_15m[i] > 65
        
        # Bollinger Band regime detection
        bbw_squeeze = bbw_pct[i] < 0.3  # BBW in bottom 30% of range
        bbw_expanding = bbw[i] > bbw[i-5] if i > 5 else False
        bbw_breakout = bbw_squeeze and bbw_expanding
        
        # Volume confirmation
        volume_confirmed = vol_ratio[i] > 1.2
        
        # Daily trend (local)
        trend_15m_bullish = hma_20[i] > hma_50[i]
        trend_15m_bearish = hma_20[i] < hma_50[i]
        
        # Momentum
        mom_strong_pos = momentum[i] > 2.0
        mom_strong_neg = momentum[i] < -2.0
        
        new_signal = 0.0
        
        # LONG ENTRY: 4h bullish + 1h RSI pullback + 15m confirmation
        if trend_4h_bullish and rsi_1h_oversold:
            # Entry when 15m RSI oversold + volume + momentum
            if rsi_15m_oversold and volume_confirmed:
                new_signal = SIZE_ENTRY
            # Or breakout from BB squeeze
            elif bbw_breakout and close[i] > bb_upper[i] and volume_confirmed:
                new_signal = SIZE_ENTRY
            # Or trend continuation with neutral RSI
            elif trend_15m_bullish and rsi_1h_neutral and mom_strong_pos:
                new_signal = SIZE_ENTRY
        
        # SHORT ENTRY: 4h bearish + 1h RSI pullback + 15m confirmation
        elif trend_4h_bearish and rsi_1h_overbought:
            # Entry when 15m RSI overbought + volume + momentum
            if rsi_15m_overbought and volume_confirmed:
                new_signal = -SIZE_ENTRY
            # Or breakdown from BB squeeze
            elif bbw_breakout and close[i] < bb_lower[i] and volume_confirmed:
                new_signal = -SIZE_ENTRY
            # Or trend continuation with neutral RSI
            elif trend_15m_bearish and rsi_1h_neutral and mom_strong_neg:
                new_signal = -SIZE_ENTRY
        
        # Stoploss logic (Rule 6) - check BEFORE updating position tracking
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