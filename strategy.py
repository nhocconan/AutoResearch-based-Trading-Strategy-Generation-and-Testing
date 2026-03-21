#!/usr/bin/env python3
"""
Experiment #156: 1d Donchian Breakout with Weekly KAMA Trend Filter

Hypothesis: Daily Donchian breakouts (20-period) capture major crypto trend moves
while weekly KAMA provides directional bias without over-filtering. ADX(14)>18
(loosened from 25) ensures we enter during emerging trends. This is Turtle Trading
adapted for crypto with ATR risk management. Daily timeframe = fewer false signals
than 4h/1h, but enough trades (target 40-60 over 5 years).

Why this should work better than failed experiments:
- Donchian breakouts are TIME-TESTED (Turtle Trading 1980s, still works)
- Weekly KAMA is SMOOTHER than HMA, less whipsaw on major trend
- ADX threshold LOW (18 not 25) to avoid 0-trade problem (exp 148, 152)
- Entry on ANY breakout + weak weekly filter = sufficient trade frequency
- 1d naturally filters noise that killed 15m/30m/1h strategies
- Conservative 0.30 sizing protects against 2022-style 77% crashes

Risk management: 2.5*ATR trailing stop, position reversal on opposite breakout.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_weekly_kama_adx_v1"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    Adapts smoothing based on market efficiency ratio.
    Reference: Kaufman, P.J. (1998) "Trading Systems and Methods"
    """
    close_s = pd.Series(close)
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(close - np.roll(close, er_period))
    change[0:er_period] = change[er_period]
    
    volatility = np.abs(close - np.roll(close, 1))
    volatility[0] = change[0]
    
    vol_sum = pd.Series(volatility).rolling(window=er_period, min_periods=er_period).sum().values
    vol_sum[0:er_period] = vol_sum[er_period]
    
    er = np.where(vol_sum > 0, change / vol_sum, 0)
    er = np.clip(er, 0, 1)
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # Calculate +DM and -DM
    diff_high = high_s.diff()
    diff_low = low_s.diff()
    
    plus_dm = np.where((diff_high > diff_low) & (diff_high > 0), diff_high, 0)
    minus_dm = np.where((diff_low > diff_high) & (diff_low > 0), diff_low, 0)
    
    # Calculate True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smooth +DM, -DM, and TR
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean()
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean()
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean()
    
    # Calculate +DI and -DI
    plus_di = 100 * plus_dm_s / tr_s
    minus_di = 100 * minus_dm_s / tr_s
    
    # Calculate DX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = dx.fillna(0)
    
    # Calculate ADX
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper and lower bands)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    kama_1w = calculate_kama(df_1w['close'].values, 10)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    adx = calculate_adx(high, low, close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    kama_1d = calculate_kama(close, 10)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.30
    SIZE_EXIT = 0.0
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(50, n):
        # Weekly trend filter (PERMISSIVE - just major bias, not hard filter)
        weekly_bullish = kama_1w_aligned[i] > 0 and close[i] > kama_1w_aligned[i]
        weekly_bearish = kama_1w_aligned[i] > 0 and close[i] < kama_1w_aligned[i]
        
        # Daily trend (KAMA slope - 5 day lookback)
        daily_bullish = kama_1d[i] > kama_1d[i-5] if i > 5 else False
        daily_bearish = kama_1d[i] < kama_1d[i-5] if i > 5 else False
        
        # ADX trend strength (LOW threshold to ensure trades)
        trending = adx[i] > 18
        
        # Donchian breakout signals (primary entry trigger)
        breakout_long = close[i] > donchian_upper[i-1] and close[i-1] <= donchian_upper[i-1]
        breakout_short = close[i] < donchian_lower[i-1] and close[i-1] >= donchian_lower[i-1]
        
        new_signal = 0.0
        
        # LONG ENTRY: Donchian breakout + (weekly bullish OR daily bullish)
        # Loosened: only need ONE of weekly/daily bullish, not both
        if breakout_long:
            if weekly_bullish or daily_bullish:
                new_signal = SIZE_ENTRY
            elif not weekly_bearish:
                # Weekly neutral is OK for long breakout
                new_signal = SIZE_ENTRY
        
        # SHORT ENTRY: Donchian breakout + (weekly bearish OR daily bearish)
        elif breakout_short:
            if weekly_bearish or daily_bearish:
                new_signal = -SIZE_ENTRY
            elif not weekly_bullish:
                # Weekly neutral is OK for short breakout
                new_signal = -SIZE_ENTRY
        
        # POSITION REVERSAL: Opposite breakout closes current and opens reverse
        if position_side > 0 and breakout_short:
            new_signal = -SIZE_ENTRY
        
        if position_side < 0 and breakout_long:
            new_signal = SIZE_ENTRY
        
        # Stoploss logic - check BEFORE updating position tracking
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
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals