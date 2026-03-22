#!/usr/bin/env python3
"""
Experiment #240: 1d Z-Score Mean Reversion with Trend Filter and Volume Confirmation

Hypothesis: On daily timeframe, crypto exhibits mean-reverting behavior around 
longer-term moving averages, especially after extended moves. Using Z-score to 
measure deviation from SMA(50) combined with RSI and volume confirmation can 
capture reversals while avoiding counter-trend traps.

Why this might work for 1d:
- Daily bars filter out noise present in lower timeframes
- Z-score > 2.0 or < -2.0 captures extreme deviations (statistically significant)
- SMA(200) provides long-term trend bias filter
- Volume spike confirms genuine reversal interest
- ATR trailing stop protects against continued trends
- Conservative sizing (0.25) controls drawdown in volatile crypto

Key differences from failed strategies:
- #228, #234 (Donchian breakout): Pure trend-following fails in range markets
- #232, #237 (KAMA + RSI): Too many whipsaws without Z-score filter
- This uses statistical deviation (Z-score) rather than fixed thresholds
- Volume confirmation reduces false signals

Timeframe: 1d (REQUIRED for this experiment)
HTF: None (single TF as per experiment rules)
Position sizing: 0.25 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd

name = "1d_zscore_meanrev_sma_rsi_volume_atr_v1"
timeframe = "1d"
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

def calculate_zscore(close, period=50):
    """Calculate Z-score of price relative to rolling mean."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    zscore = (close_s - sma) / (std + 1e-10)
    return zscore.values

def calculate_rsi(close, period=14):
    """Calculate RSI (Relative Strength Index)."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

def calculate_sma(close, period):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    return sma.values

def calculate_volume_zscore(volume, period=20):
    """Calculate Z-score of volume relative to rolling mean."""
    vol_s = pd.Series(volume)
    vol_mean = vol_s.rolling(window=period, min_periods=period).mean()
    vol_std = vol_s.rolling(window=period, min_periods=period).std()
    vol_zscore = (vol_s - vol_mean) / (vol_std + 1e-10)
    return vol_zscore.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    adx = np.zeros(n)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        # True Range
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Directional Movement
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(0, high[i] - high[i-1])
        else:
            plus_dm[i] = 0
            
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(0, low[i-1] - low[i])
        else:
            minus_dm[i] = 0
    
    # Smooth TR, +DM, -DM using Wilder's method
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate +DI, -DI
    plus_di = 100 * plus_dm_smooth / (tr_smooth + 1e-10)
    minus_di = 100 * minus_dm_smooth / (tr_smooth + 1e-10)
    
    # Calculate DX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    
    # Smooth DX to get ADX
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Calculate all indicators (vectorized where possible)
    atr = calculate_atr(high, low, close, 14)
    zscore_50 = calculate_zscore(close, 50)
    rsi_14 = calculate_rsi(close, 14)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    vol_zscore = calculate_volume_zscore(volume, 20)
    adx_14 = calculate_adx(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_HALF = 0.125
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    entry_atr = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(zscore_50[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(sma_200[i]) or np.isnan(adx_14[i]):
            signals[i] = 0.0
            continue
        
        # === LONG-TERM TREND BIAS ===
        # Price above SMA200 = bullish bias (prefer longs)
        # Price below SMA200 = bearish bias (prefer shorts)
        bull_trend = close[i] > sma_200[i]
        bear_trend = close[i] < sma_200[i]
        
        # === MEAN REVERSION SIGNALS ===
        # Z-score < -1.5 = price significantly below mean (oversold)
        # Z-score > +1.5 = price significantly above mean (overbought)
        # Relaxed from -2.0/+2.0 to ensure more trades
        zscore_oversold = zscore_50[i] < -1.5
        zscore_overbought = zscore_50[i] > 1.5
        
        # RSI confirmation (less extreme than typical for more trades)
        rsi_oversold = rsi_14[i] < 40
        rsi_overbought = rsi_14[i] > 60
        
        # Volume spike confirmation (z-score > 1.0 = above average volume)
        volume_spike = vol_zscore[i] > 1.0
        
        # ADX filter - avoid entering when trend is very strong (ADX > 35)
        # Prefer mean reversion when ADX < 30 (weaker trend)
        weak_trend = adx_14[i] < 30
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Z-score oversold + RSI oversold + (volume spike OR weak trend)
        # Prefer longs when price > SMA200 (bullish bias)
        if zscore_oversold and rsi_oversold:
            if volume_spike or weak_trend:
                if bull_trend or not bear_trend:
                    new_signal = SIZE_BASE
        
        # === SHORT ENTRY ===
        # Z-score overbought + RSI overbought + (volume spike OR weak trend)
        # Prefer shorts when price < SMA200 (bearish bias)
        if zscore_overbought and rsi_overbought:
            if volume_spike or weak_trend:
                if bear_trend or not bull_trend:
                    new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        # Check stoploss on EXISTING position before considering new entry
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                # Trailing stop: 2.5 * ATR below highest close
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                # Trailing stop: 2.5 * ATR above lowest close
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
        
        # === TAKE PROFIT: Reduce to half at 2R profit ===
        if in_position and new_signal != 0.0 and position_side != 0:
            if position_side > 0:
                profit = close[i] - entry_price
                if profit >= 2.0 * entry_atr:
                    new_signal = SIZE_HALF  # Take partial profit
            if position_side < 0:
                profit = entry_price - close[i]
                if profit >= 2.0 * entry_atr:
                    new_signal = -SIZE_HALF  # Take partial profit
        
        # === UPDATE POSITION TRACKING FOR NEXT BAR ===
        if new_signal != 0.0:
            if not in_position:
                # Entering new position
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_atr = atr[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_atr = atr[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            # else: maintaining same position direction (possibly reduced size)
        else:
            # Exiting position (signal-based or stoploss)
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals