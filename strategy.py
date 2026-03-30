#!/usr/bin/env python3
"""
Experiment #022: 1d Donchian Breakout + Weekly Trend + Volume (1d)

HYPOTHESIS: Simple price channel breakout on daily with weekly trend confirmation
will capture major trends while avoiding whipsaws in range-bound markets.

WHY IT SHOULD WORK IN BOTH BULL AND BEAR:
- Bull (2021, 2023-2024): Breakouts above Donchian(20) + volume spike + weekly bull = strong long runs
- Bear (2022): Weekly trend filter prevents chasing broken breakouts; price oscillates within range
- Weekly MA direction (8-period) gives 1-week lookback = smooth, not choppy

KEY INSIGHTS FROM DATABASE:
- Donchian(20) breakout + volume confirmation = proven pattern (test Sharpe 1.1-1.5)
- Weekly trend filter = key for avoiding bear market whipsaws
- ATR stoploss = mandatory for risk management

TARGET: 100-150 total trades over 4 years (25-37/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_weekly_vol_v1"
timeframe = "1d"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range - vectorized"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Donchian Channel - vectorized"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    middle = pd.Series((pd.Series(high).rolling(window=period, min_periods=period).max() + 
                        pd.Series(low).rolling(window=period, min_periods=period).min()) / 2).values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, middle, lower

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - vectorized"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    di_plus = np.zeros(n)
    di_minus = np.zeros(n)
    dx = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 0:
            di_plus[i] = 100 * plus_dm_smooth[i] / atr[i]
            di_minus[i] = 100 * minus_dm_smooth[i] / atr[i]
            di_sum = di_plus[i] + di_minus[i]
            if di_sum > 0:
                dx[i] = 100 * abs(di_plus[i] - di_minus[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_weekly = get_htf_data(prices, '1w')
    
    # Weekly indicators
    weekly_ma = pd.Series(df_weekly['close'].values).rolling(window=8, min_periods=8).mean().values
    weekly_close = df_weekly['close'].values
    
    # Weekly trend: price above/below 8-period MA
    weekly_trend_up = weekly_close > weekly_ma
    weekly_trend_down = weekly_close < weekly_ma
    
    # Align to daily timeframe
    weekly_trend_up_aligned = align_htf_to_ltf(prices, df_weekly, weekly_trend_up.astype(float))
    weekly_trend_down_aligned = align_htf_to_ltf(prices, df_weekly, weekly_trend_down.astype(float))
    
    # === Local 1d indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_middle, donchian_lower = calculate_donchian(high, low, period=20)
    adx = calculate_adx(high, low, close, period=14)
    
    # Volume ratio
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    trailing_high = 0.0
    trailing_low = 0.0
    
    warmup = 60  # Donchian(20) + volume(20) + weekly(8)
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(weekly_trend_up_aligned[i]):
            signals[i] = 0.0
            continue
        
        # HTF trend
        htf_bull = weekly_trend_up_aligned[i] > 0.5
        htf_bear = weekly_trend_down_aligned[i] > 0.5
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # Trend strength
        strong_trend = adx[i] > 22
        
        desired_signal = 0.0
        
        if not in_position:
            # === LONG ENTRY ===
            # Price breaks above Donchian(20) high + pullback confirmation
            # 1. Close above upper band (breakout)
            # 2. Currently above middle (not immediately falling back)
            # 3. Volume spike confirmation
            # 4. Trend strength (ADX > 22)
            # 5. Weekly trend bull OR neutral (not required but preferred)
            bullish_breakout = (close[i] > donchian_upper[i-1] if i > 0 else False)
            price_above_mid = close[i] > donchian_middle[i]
            
            if bullish_breakout and price_above_mid and vol_spike and strong_trend:
                # Weekly bull OR neutral (not bear)
                if htf_bull or not htf_bear:
                    desired_signal = SIZE
            
            # === SHORT ENTRY ===
            # Price breaks below Donchian(20) low + pullback confirmation
            bearish_breakout = (close[i] < donchian_lower[i-1] if i > 0 else False)
            price_below_mid = close[i] < donchian_middle[i]
            
            if bearish_breakout and price_below_mid and vol_spike and strong_trend:
                # Weekly bear OR neutral (not bull)
                if htf_bear or not htf_bull:
                    desired_signal = -SIZE
        
        # === EXIT MANAGEMENT ===
        if in_position:
            if position_side > 0:
                # Update trailing high
                if i == entry_bar or high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Trailing stop: 2.5 ATR from highest high since entry
                stop_price = trailing_high - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Exit if weekly turns bear (contradicts position)
                if htf_bear:
                    desired_signal = 0.0
                
                # Exit if trend weakens severely
                if adx[i] < 16:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Update trailing low
                if i == entry_bar or low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Trailing stop: 2.5 ATR from lowest low since entry
                stop_price = trailing_low + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Exit if weekly turns bull (contradicts position)
                if htf_bull:
                    desired_signal = 0.0
                
                # Exit if trend weakens severely
                if adx[i] < 16:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 3 bars to reduce fee churn ===
        if in_position and (i - entry_bar) < 3:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                trailing_high = high[i]
                trailing_low = low[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals