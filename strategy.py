#!/usr/bin/env python3
"""
Experiment #299: 12h Donchian Breakout + Daily/Weekly HMA Trend + Volume/ADX Filter
Hypothesis: 12h Donchian breakouts capture major trend moves with fewer trades (lower fee drag).
Daily HMA provides intermediate trend filter, Weekly HMA provides macro bias.
Volume spike (1.5x avg) confirms breakout validity. ADX>25 ensures trending regime (avoid range whipsaw).
ATR trailing stops (2.5*ATR) protect capital. Position size 0.25 balances risk/reward.
Target: Beat Sharpe=0.499 from current best while ensuring >=10 trades per symbol on 12h timeframe.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_daily_weekly_hma_volume_adx_atr_v1"
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
    return upper, lower

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean() / pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean()
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean() / pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean()
    
    dx = 100 * np.abs(plus_di - minus_di) / (np.abs(plus_di + minus_di) + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_volume_sma(volume, period=20):
    """Calculate volume SMA for volume spike detection."""
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    adx = calculate_adx(high, low, close, 14)
    volume_sma = calculate_volume_sma(volume, 20)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # Track previous values for breakout detection
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    prev_high = np.roll(high, 1)
    prev_high[0] = high[0]
    prev_low = np.roll(low, 1)
    prev_low[0] = low[0]
    
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
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(atr[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        # Weekly macro trend bias
        weekly_bullish = close[i] > hma_1w_aligned[i]
        weekly_bearish = close[i] < hma_1w_aligned[i]
        
        # Daily trend filter
        daily_bullish = close[i] > hma_1d_aligned[i]
        daily_bearish = close[i] < hma_1d_aligned[i]
        
        # Trending regime filter (ADX > 25)
        trending = adx[i] > 25
        
        # Volume spike confirmation (1.5x average)
        volume_spike = volume[i] > 1.5 * volume_sma[i]
        
        # Donchian breakout signals
        breakout_long = prev_close[i] <= donchian_upper[i] and close[i] > donchian_upper[i]
        breakout_short = prev_close[i] >= donchian_lower[i] and close[i] < donchian_lower[i]
        
        # Alternative: price near upper/lower band (momentum continuation)
        near_upper = close[i] > donchian_upper[i] - 0.5 * atr[i]
        near_lower = close[i] < donchian_lower[i] + 0.5 * atr[i]
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Primary: Weekly bullish + Daily bullish + Donchian breakout + Volume spike + ADX trending
        if weekly_bullish and daily_bullish and breakout_long and volume_spike and trending:
            new_signal = SIZE_ENTRY
        # Secondary: Weekly bullish + Daily bullish + Donchian breakout + ADX trending (no volume req)
        elif weekly_bullish and daily_bullish and breakout_long and trending:
            new_signal = SIZE_ENTRY
        # Tertiary: Weekly bullish + Price > Donchian upper + ADX trending (momentum continuation)
        elif weekly_bullish and daily_bullish and near_upper and trending and close[i] > prev_close[i]:
            new_signal = SIZE_ENTRY
        # Quaternary: Daily bullish + Donchian breakout + Volume spike (simpler for more trades)
        elif daily_bullish and breakout_long and volume_spike:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRY ===
        # Primary: Weekly bearish + Daily bearish + Donchian breakout + Volume spike + ADX trending
        if weekly_bearish and daily_bearish and breakout_short and volume_spike and trending:
            new_signal = -SIZE_ENTRY
        # Secondary: Weekly bearish + Daily bearish + Donchian breakout + ADX trending (no volume req)
        elif weekly_bearish and daily_bearish and breakout_short and trending:
            new_signal = -SIZE_ENTRY
        # Tertiary: Weekly bearish + Price < Donchian lower + ADX trending (momentum continuation)
        elif weekly_bearish and daily_bearish and near_lower and trending and close[i] < prev_close[i]:
            new_signal = -SIZE_ENTRY
        # Quaternary: Daily bearish + Donchian breakout + Volume spike (simpler for more trades)
        elif daily_bearish and breakout_short and volume_spike:
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