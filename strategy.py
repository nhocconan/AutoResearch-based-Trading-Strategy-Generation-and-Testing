#!/usr/bin/env python3
"""
Experiment #004: 1d Donchian + 1w SMA50 + Volume Confirmation

HYPOTHESIS: Simple 1d Donchian breakout with weekly trend confirmation and volume validation.

WHY IT SHOULD WORK IN BOTH BULL AND BEAR:
- Bull: Price breaks above 20d high + above 1w SMA50 + volume spike = strong long
- Bear: Price breaks below 20d low + below 1w SMA50 + volume spike = strong short
- Range (2022 bottom): 1w SMA filter avoids catching falling knife
- ATR stoploss manages risk in both directions

PATTERN: Follows proven DB winner: mtf_4h_chop_donchian_vol_regime (Sharpe 1.49)
Adjusted for 1d timeframe: Donchian + volume + ATR stoploss.

TARGET: 30-80 trades over 4 years (7-20/year) - enough for statistical validity.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_1w_sma_vol_v1"
timeframe = "1d"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_adx(high, low, close, period=14):
    """Average Directional Index"""
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
    
    # === Load HTF data ONCE before loop ===
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly SMA50 for trend direction
    sma_1w = pd.Series(df_1w['close'].values).rolling(window=50, min_periods=50).mean().values
    sma_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_1w)
    
    # Weekly EMA21 for faster confirmation
    ema_1w = pd.Series(df_1w['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # === Local 1d indicators ===
    # Donchian Channel (20) - standard breakout
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR for stoploss
    atr = calculate_atr(high, low, close, period=14)
    
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # ADX for regime (trend strength)
    adx = calculate_adx(high, low, close, period=14)
    
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
    
    warmup = 100  # Need at least 50 for weekly SMA, 20 for Donchian, 14 for ATR
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        if np.isnan(sma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === HTF TREND (1w) ===
        htf_bull = close[i] > sma_1w_aligned[i]
        htf_bear = close[i] < sma_1w_aligned[i]
        
        # Weekly momentum (fast EMA vs slow SMA)
        weekly_momentum = ema_1w_aligned[i] > sma_1w_aligned[i] if not np.isnan(ema_1w_aligned[i]) else htf_bull
        
        # === DONCHIAN BREAKOUT ===
        # Check for breakout on current bar
        bullish_breakout = close[i] > donchian_upper[i]
        prev_above_upper = close[i-1] > donchian_upper[i-1] if i > 0 else False
        new_bullish_breakout = bullish_breakout and not prev_above_upper
        
        bearish_breakout = close[i] < donchian_lower[i]
        prev_below_lower = close[i-1] < donchian_lower[i-1] if i > 0 else False
        new_bearish_breakout = bearish_breakout and not prev_below_lower
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.4  # Volume 40% above average
        
        # === REGIME FILTER (ADX) ===
        strong_trend = adx[i] > 22  # Moderate threshold
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG: Bullish breakout + volume + HTF bull
            if bullish_breakout and vol_spike and htf_bull and strong_trend:
                desired_signal = SIZE
            
            # SHORT: Bearish breakout + volume + HTF bear
            elif bearish_breakout and vol_spike and htf_bear and strong_trend:
                desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR trailing stop) ===
        if in_position:
            if position_side > 0:
                # Update trailing high
                if i == entry_bar or high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Trailing stop
                stop_price = trailing_high - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Exit if HTF turns bearish
                if htf_bear:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Update trailing low
                if i == entry_bar or low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Trailing stop
                stop_price = trailing_low + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Exit if HTF turns bullish
                if htf_bull:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 2 bars to reduce fee churn ===
        if in_position and (i - entry_bar) < 2:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                entry_bar = i
                trailing_high = high[i]
                trailing_low = low[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals