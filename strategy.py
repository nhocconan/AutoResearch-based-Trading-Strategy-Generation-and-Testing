#!/usr/bin/env python3
"""
EXPERIMENT #004 - Donchian Breakout + Daily Trend Filter Strategy (4h)
=======================================================================
Hypothesis: Donchian Channel breakouts capture momentum moves well on 4h timeframe.
Combined with 1d HMA for major trend direction filter and ADX for trend strength,
this should capture sustained trending moves while avoiding counter-trend traps.

Key differences from failed experiments:
- 4h primary timeframe (this experiment's rotation)
- 1d HMA for major trend filter (higher TF = more reliable trend)
- Donchian(20) breakouts for clean entry signals
- Volume confirmation to filter false breakouts
- ATR-based stoploss at 2.5*ATR (wider for 4h timeframe)
- Conservative position size (0.30 max) for drawdown control
- Fixed signal values to avoid read-only array issues

Primary TF: 4h | HTF: 1d HMA trend filter
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "donchian_daily_trend_4h_v1"
timeframe = "4h"
leverage = 1.0


def calculate_hma(close: np.ndarray, period: int = 21) -> np.ndarray:
    """
    Calculate Hull Moving Average (HMA)
    HMA = WMA(2*WMA(n/2) - WMA(n)) with sqrt(n) period
    More responsive than EMA with less lag
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    close_s = pd.Series(close)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half_period, min_periods=half_period, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    hull_input = 2 * wma_half - wma_full
    hma = hull_input.ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    
    hma_values = hma.values
    hma_values[:period] = np.nan
    return hma_values


def calculate_donchian(high: np.ndarray, low: np.ndarray, period: int = 20) -> tuple:
    """
    Calculate Donchian Channel (upper, lower, middle)
    Upper = highest high over period
    Lower = lowest low over period
    Middle = (upper + lower) / 2
    """
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max().values
    lower = low_s.rolling(window=period, min_periods=period).min().values
    middle = (upper + lower) / 2
    
    upper[:period] = np.nan
    lower[:period] = np.nan
    middle[:period] = np.nan
    
    return upper, lower, middle


def calculate_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calculate Average Directional Index (ADX)
    ADX > 25 indicates strong trend, ADX < 20 indicates range
    """
    n = len(close)
    if n < period * 2:
        return np.full(n, np.nan)
    
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
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    # Smoothed values
    atr = tr.rolling(window=period, min_periods=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period, min_periods=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period, min_periods=period).mean() / atr)
    
    # DX and ADX
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.rolling(window=period, min_periods=period).mean()
    
    adx_values = adx.values
    adx_values[:period * 2] = np.nan
    return adx_values


def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate ATR with proper min_periods"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    atr[:period] = np.nan
    return atr


def calculate_volume_sma(volume: np.ndarray, period: int = 20) -> np.ndarray:
    """Calculate volume SMA for volume confirmation"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_s = pd.Series(volume)
    vol_sma = vol_s.rolling(window=period, min_periods=period).mean().values
    vol_sma[:period] = np.nan
    return vol_sma


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    # Extract price data
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # === LOAD HTF DATA ONCE BEFORE LOOP (CRITICAL RULE #1) ===
    df_1d = get_htf_data(prices, '1d')
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)  # auto shift(1)
    
    # === CALCULATE 4h INDICATORS (vectorized before loop) ===
    donchian_upper, donchian_lower, donchian_middle = calculate_donchian(high, low, period=20)
    adx = calculate_adx(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    volume_sma = calculate_volume_sma(volume, period=20)
    
    # Also calculate 4h HMA for additional trend confirmation
    hma_4h = calculate_hma(close, period=21)
    
    # === SIGNAL PARAMETERS ===
    SIZE_ENTRY = 0.30      # 30% position on entry
    SIZE_HALF = 0.15       # 15% after take profit
    STOPLOSS_MULT = 2.5    # 2.5*ATR stoploss (wider for 4h)
    TAKEPROFIT_MULT = 2.0  # 2R take profit
    ADX_THRESHOLD = 25     # Only trade when ADX > 25 (strong trend)
    VOLUME_MULT = 1.2      # Volume must be > 1.2x average for confirmation
    
    signals = np.zeros(n)
    
    # Track position state for stoploss/takeprofit
    entry_price = 0.0
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    trailing_stop = 0.0
    
    min_lookback = 100  # Ensure all indicators are valid
    
    for i in range(min_lookback, n):
        # Skip if any indicator is NaN
        if (np.isnan(hma_1d_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(adx[i]) or 
            np.isnan(atr[i]) or np.isnan(volume_sma[i]) or np.isnan(hma_4h[i])):
            signals[i] = 0.0
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            continue
        
        current_atr = atr[i]
        current_price = close[i]
        current_adx = adx[i]
        current_hma_1d = hma_1d_aligned[i]
        current_hma_4h = hma_4h[i]
        current_volume = volume[i]
        current_volume_sma = volume_sma[i]
        
        upper_breakout = donchian_upper[i]
        lower_breakout = donchian_lower[i]
        
        # === TREND FILTER (1d HMA) ===
        # Price above 1d HMA = major uptrend, below = major downtrend
        major_trend_up = current_price > current_hma_1d
        major_trend_down = current_price < current_hma_1d
        
        # === TREND STRENGTH FILTER (ADX) ===
        strong_trend = current_adx > ADX_THRESHOLD
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = current_volume > (current_volume_sma * VOLUME_MULT)
        
        # === 4h TREND CONFIRMATION ===
        trend_4h_up = current_price > current_hma_4h
        trend_4h_down = current_price < current_hma_4h
        
        # === ENTRY SIGNALS ===
        new_signal = 0.0
        
        if position_side == 0:
            # === LONG ENTRY: major uptrend + 4h uptrend + ADX strong + volume + breakout ===
            if (major_trend_up and trend_4h_up and strong_trend and 
                volume_confirmed and current_price > upper_breakout):
                # Check previous bar didn't already breakout (avoid chasing)
                if i > min_lookback and close[i-1] <= donchian_upper[i-1]:
                    new_signal = SIZE_ENTRY
                    entry_price = current_price
                    position_side = 1
                    highest_since_entry = current_price
                    lowest_since_entry = current_price
                    trailing_stop = entry_price - STOPLOSS_MULT * current_atr
            
            # === SHORT ENTRY: major downtrend + 4h downtrend + ADX strong + volume + breakout ===
            elif (major_trend_down and trend_4h_down and strong_trend and 
                  volume_confirmed and current_price < lower_breakout):
                # Check previous bar didn't already breakout (avoid chasing)
                if i > min_lookback and close[i-1] >= donchian_lower[i-1]:
                    new_signal = -SIZE_ENTRY
                    entry_price = current_price
                    position_side = -1
                    highest_since_entry = current_price
                    lowest_since_entry = current_price
                    trailing_stop = entry_price + STOPLOSS_MULT * current_atr
        
        elif position_side == 1:
            # Track highest price since entry for trailing
            highest_since_entry = max(highest_since_entry, current_price)
            
            # === TRAILING STOP: move stop up as price rises ===
            new_trailing_stop = highest_since_entry - STOPLOSS_MULT * current_atr
            if new_trailing_stop > trailing_stop:
                trailing_stop = new_trailing_stop
            
            # === STOPLOSS: price drops below trailing stop ===
            if current_price < trailing_stop:
                new_signal = 0.0
                position_side = 0
                entry_price = 0.0
                trailing_stop = 0.0
            
            # === TAKE PROFIT: at 2R, reduce to half ===
            elif entry_price > 0:
                profit_r = (current_price - entry_price) / current_atr
                if profit_r >= TAKEPROFIT_MULT:
                    new_signal = SIZE_HALF
                    # Update trailing stop to lock in profit
                    trailing_stop = entry_price + 0.5 * current_atr
            
            # === EXIT: Major trend reversal (1d HMA) ===
            elif major_trend_down:
                new_signal = 0.0
                position_side = 0
                entry_price = 0.0
                trailing_stop = 0.0
            
            # === EXIT: 4h trend reversal ===
            elif trend_4h_down:
                new_signal = 0.0
                position_side = 0
                entry_price = 0.0
                trailing_stop = 0.0
            
            else:
                # Maintain position
                new_signal = SIZE_ENTRY if new_signal == 0 else new_signal
        
        elif position_side == -1:
            # Track lowest price since entry for trailing
            lowest_since_entry = min(lowest_since_entry, current_price)
            
            # === TRAILING STOP: move stop down as price falls ===
            new_trailing_stop = lowest_since_entry + STOPLOSS_MULT * current_atr
            if trailing_stop == 0 or new_trailing_stop < trailing_stop:
                trailing_stop = new_trailing_stop
            
            # === STOPLOSS: price rises above trailing stop ===
            if current_price > trailing_stop:
                new_signal = 0.0
                position_side = 0
                entry_price = 0.0
                trailing_stop = 0.0
            
            # === TAKE PROFIT: at 2R, reduce to half ===
            elif entry_price > 0:
                profit_r = (entry_price - current_price) / current_atr
                if profit_r >= TAKEPROFIT_MULT:
                    new_signal = -SIZE_HALF
                    # Update trailing stop to lock in profit
                    trailing_stop = entry_price - 0.5 * current_atr
            
            # === EXIT: Major trend reversal (1d HMA) ===
            elif major_trend_up:
                new_signal = 0.0
                position_side = 0
                entry_price = 0.0
                trailing_stop = 0.0
            
            # === EXIT: 4h trend reversal ===
            elif trend_4h_up:
                new_signal = 0.0
                position_side = 0
                entry_price = 0.0
                trailing_stop = 0.0
            
            else:
                # Maintain position
                new_signal = -SIZE_ENTRY if new_signal == 0 else new_signal
        
        signals[i] = new_signal
    
    return signals