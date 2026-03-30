#!/usr/bin/env python3
"""
Experiment #022: 6h Donchian Breakout + 1d Trend + Volume Confirmation

HYPOTHESIS: Simple but effective trend-following system using proven components:
1. 6h Donchian(20) breakout - price structure
2. 1d HTF trend direction - multi-timeframe confirmation
3. Volume spike - momentum validation
4. ADX filter - avoid ranging markets

WHY IT SHOULD WORK:
- Bull: Price breaks 20-period high on 6h + 1d trend up + volume spike + ADX>25
- Bear: Price breaks 20-period low on 6h + 1d trend down + volume spike + ADX>25
- Range: ADX<25 = no trades (avoids whipsaws)

KEY INSIGHT: Previous strategies FAILED with too_few_trades due to too many filters.
This uses ONLY 4 conditions: Donchian break + HTF direction + volume + ADX.
Target: 75-150 total trades over 4 years (15-30/year) = enough for stats, not overtrading.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_donchian_1d_trend_vol_v1"
timeframe = "6h"
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
    """
    Average Directional Index - trend strength
    ADX > 25 = trending, ADX < 20 = ranging
    """
    n = len(close)
    if n < period + 2:
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low"""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load 1d HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # === 1d HTF: Simple SMA(50) for trend direction ===
    sma_50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    htf_trend_up = close_1d > sma_50_1d
    htf_trend_down = close_1d < sma_50_1d
    
    # Align HTF to 6h
    htf_up_aligned = align_htf_to_ltf(prices, df_1d, htf_trend_up.astype(float))
    htf_down_aligned = align_htf_to_ltf(prices, df_1d, htf_trend_down.astype(float))
    
    # === Local 6h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    adx = calculate_adx(high, low, close, period=14)
    
    # Donchian(20)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Volume: 20-bar MA ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # === Parameters ===
    SIZE = 0.28
    DONCHIAN_PERIOD = 20
    ADX_THRESHOLD = 25
    VOL_SPIKE_THRESHOLD = 1.4
    ATR_STOP_MULT = 2.5
    
    # Signals array
    signals = np.zeros(n)
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    trailing_high = 0.0
    trailing_low = 0.0
    
    warmup = 100  # 6h * 100 = 600 bars = 25+ days, enough for indicators
    
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
        
        # HTF trend
        htf_up = htf_up_aligned[i] > 0.5 if not np.isnan(htf_up_aligned[i]) else False
        htf_down = htf_down_aligned[i] > 0.5 if not np.isnan(htf_down_aligned[i]) else False
        
        # Donchian breakout signals
        bullish_breakout = high[i] > donchian_upper[i] and high[i-1] <= donchian_upper[i-1]
        bearish_breakout = low[i] < donchian_lower[i] and low[i-1] >= donchian_lower[i-1]
        
        # Conditions
        vol_spike = vol_ratio[i] > VOL_SPIKE_THRESHOLD
        strong_trend = adx[i] > ADX_THRESHOLD
        moderate_trend = adx[i] > 20
        
        desired_signal = 0.0
        
        # === ENTRY LOGIC ===
        if not in_position:
            # LONG: Bullish Donchian break + 1d uptrend + volume + trend strength
            if bullish_breakout and htf_up and (vol_spike or strong_trend):
                desired_signal = SIZE
            
            # SHORT: Bearish Donchian break + 1d downtrend + volume + trend strength
            elif bearish_breakout and htf_down and (vol_spike or strong_trend):
                desired_signal = -SIZE
        
        # === STOPLOSS AND EXIT ===
        else:
            if position_side > 0:
                # Update trailing high
                if high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Trailing stop (2.5 ATR from highest point since entry)
                stop_price = trailing_high - ATR_STOP_MULT * entry_atr
                
                if low[i] < stop_price:
                    desired_signal = 0.0
                elif htf_down:  # Exit if HTF trend reverses
                    desired_signal = 0.0
                else:
                    desired_signal = SIZE
            
            elif position_side < 0:
                # Update trailing low
                if low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Trailing stop
                stop_price = trailing_low + ATR_STOP_MULT * entry_atr
                
                if high[i] > stop_price:
                    desired_signal = 0.0
                elif htf_up:  # Exit if HTF trend reverses
                    desired_signal = 0.0
                else:
                    desired_signal = -SIZE
        
        # === MINIMUM HOLD: 6 bars (1.5 days) to avoid fee churn ===
        if in_position and (i - entry_bar) < 6:
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