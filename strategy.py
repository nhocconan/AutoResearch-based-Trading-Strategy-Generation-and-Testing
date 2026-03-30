#!/usr/bin/env python3
"""
Experiment #004: 1d Donchian Breakout + Williams %R + Weekly Trend + Volume
Primary = 1d, HTF = 1w

HYPOTHESIS: Simple but powerful - daily Donchian(20) breakout with:
1. Williams %R momentum confirmation (not overbought/oversold, just direction)
2. Weekly SMA(10) trend filter for direction bias
3. Volume spike confirmation on breakout
4. 2.5 ATR stoploss for risk management

WHY IT SHOULD WORK IN BOTH BULL AND BEAR:
- Bull: Price breaks 20d high + %R confirming + above weekly SMA = strong momentum long
- Bear: Price breaks 20d low + %R confirming + below weekly SMA = strong momentum short
- Range: Weekly filter prevents false breakouts in choppy markets
- Simple 3-condition entry = fewer trades = less fee drag

TARGET: 30-100 total over 4 years (7-25/year). HARD MAX: 150 total.
This is conservative to ensure statistical validity with only 1d bars.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_willr_weekly_vol_v1"
timeframe = "1d"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range with proper handling"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_williams_r(high, low, close, period=14):
    """Williams %R - momentum indicator"""
    n = len(close)
    willr = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        if highest_high != lowest_low:
            willr[i] = -100 * (highest_high - close[i]) / (highest_high - lowest_low)
    
    return willr

def calculate_donchian(high, low, period=20):
    """Donchian Channel - returns upper (resistance) and lower (support) bands"""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength filter"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
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
    
    # === Load 1w HTF data ONCE before loop ===
    df_1w = get_htf_data(prices, '1w')
    
    # === Weekly SMA for trend direction ===
    weekly_close = df_1w['close'].values
    weekly_sma = pd.Series(weekly_close).rolling(window=10, min_periods=10).mean().values
    weekly_above_sma = weekly_close > weekly_sma
    
    # Align weekly to daily
    weekly_above_sma_aligned = align_htf_to_ltf(prices, df_1w, weekly_above_sma.astype(float))
    
    # === Local 1d indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    willr = calculate_williams_r(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    adx = calculate_adx(high, low, close, period=14)
    
    # Volume ratio (20-day MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals array
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    trailing_high = 0.0
    trailing_low = 0.0
    last_entry_bar = -100  # Cooldown tracking
    
    warmup = 100  # Need 20 for Donchian, 14 for ATR/ADX/WillR, 20 for volume
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(willr[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(weekly_above_sma_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === ENTRY CONDITIONS ===
        desired_signal = 0.0
        
        # ADX filter - only trend in strong markets
        strong_trend = adx[i] > 22
        
        # Weekly trend alignment
        weekly_bull = weekly_above_sma_aligned[i] > 0.5
        weekly_bear = weekly_above_sma_aligned[i] < 0.5
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # Williams %R direction (not extremes, just direction)
        willr_bullish = willr[i] > -50  # Above midpoint = bullish momentum
        willr_bearish = willr[i] < -50  # Below midpoint = bearish momentum
        
        # Cooldown: at least 5 bars since last entry
        cooldown_ok = (i - last_entry_bar) >= 5
        
        if not in_position and cooldown_ok:
            # === LONG ENTRY ===
            # Breakout above 20d high + bullish %R + weekly bull + volume
            price_broke_high = close[i] > donchian_upper[i] and close[i-1] <= donchian_upper[i-1]
            
            if price_broke_high and willr_bullish and weekly_bull and vol_spike and strong_trend:
                desired_signal = SIZE
                last_entry_bar = i
            
            # === SHORT ENTRY ===
            # Breakdown below 20d low + bearish %R + weekly bear + volume
            price_broke_low = close[i] < donchian_lower[i] and close[i-1] >= donchian_lower[i-1]
            
            if price_broke_low and willr_bearish and weekly_bear and vol_spike and strong_trend:
                desired_signal = -SIZE
                last_entry_bar = i
        
        # === EXIT LOGIC ===
        if in_position:
            if position_side > 0:
                # Update trailing high
                if high[i] > trailing_high:
                    trailing_high = high[i]
                
                # 2.5 ATR trailing stop
                stop_price = trailing_high - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Exit if weekly turns bearish
                if weekly_bear:
                    desired_signal = 0.0
                
                # Exit if %R turns bearish
                if willr[i] < -60:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Update trailing low
                if low[i] < trailing_low:
                    trailing_low = low[i]
                
                # 2.5 ATR trailing stop
                stop_price = trailing_low + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Exit if weekly turns bullish
                if weekly_bull:
                    desired_signal = 0.0
                
                # Exit if %R turns bullish
                if willr[i] > -40:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 3 days to avoid fee churn ===
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