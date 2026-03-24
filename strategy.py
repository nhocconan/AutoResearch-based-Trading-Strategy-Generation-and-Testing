#!/usr/bin/env python3
"""
Experiment #851: 6h Primary + 1w/1d HTF — Donchian Trend + RSI Pullback + Volume

Hypothesis: 6h timeframe sits between 4h (too noisy) and 12h (too slow). 
Donchian channel breakouts provide clean trend signals without whipsaw.
RSI pullbacks within trend direction give optimal entry timing.
Volume confirmation reduces false breakouts common in crypto.

Key innovations:
1. 1w Donchian(20) for HTF trend bias - clean multi-week direction
2. 1d Donchian(20) for intermediate trend - confirms 1w bias
3. 6h RSI(14) pullback entries - enter on dips in uptrend, rallies in downtrend
4. Volume spike filter (1.5x 20-bar MA) - confirms breakout validity
5. ATR(14) 3.0x trailing stop - looser than 2.5x to reduce premature stop-outs
6. Discrete sizing: 0.0, ±0.25, ±0.30

Entry conditions (LOOSE to ensure ≥10 trades/train, ≥3/test):
- LONG: 1w Donchian bull + 1d Donchian bull + RSI(14) < 55 + volume spike
- SHORT: 1w Donchian bear + 1d Donchian bear + RSI(14) > 45 + volume spike

Target: Sharpe>0.45, trades>=10 train, trades>=3 test, DD>-40%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_donchian_rsi_vol_1w1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_donchian(high, low, period=20):
    """
    Donchian Channel - breakout indicator
    Upper = highest high over n periods
    Lower = lowest low over n periods
    Mid = (Upper + Lower) / 2
    """
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    mid = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    mid[:] = np.nan
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
        mid[i] = (upper[i] + lower[i]) / 2.0
    
    return upper, lower, mid

def calculate_rsi(close, period=14):
    """Relative Strength Index - momentum oscillator"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] > 1e-10:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range - volatility measure for stops"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_ma(volume, period=20):
    """Volume moving average for spike detection"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_ma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF Donchian
    donch_1w_upper_raw, donch_1w_lower_raw, donch_1w_mid_raw = calculate_donchian(
        df_1w['high'].values, df_1w['low'].values, period=20
    )
    donch_1w_upper = align_htf_to_ltf(prices, df_1w, donch_1w_upper_raw)
    donch_1w_lower = align_htf_to_ltf(prices, df_1w, donch_1w_lower_raw)
    donch_1w_mid = align_htf_to_ltf(prices, df_1w, donch_1w_mid_raw)
    
    # Calculate and align 1d Donchian
    donch_1d_upper_raw, donch_1d_lower_raw, donch_1d_mid_raw = calculate_donchian(
        df_1d['high'].values, df_1d['low'].values, period=20
    )
    donch_1d_upper = align_htf_to_ltf(prices, df_1d, donch_1d_upper_raw)
    donch_1d_lower = align_htf_to_ltf(prices, df_1d, donch_1d_lower_raw)
    donch_1d_mid = align_htf_to_ltf(prices, df_1d, donch_1d_mid_raw)
    
    # Calculate 6h indicators
    donch_6h_upper, donch_6h_lower, donch_6h_mid = calculate_donchian(high, low, period=20)
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    vol_ma_20 = calculate_volume_ma(volume, period=20)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(donch_1w_upper[i]) or np.isnan(donch_1d_upper[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1w Donchian) ===
        # Price above mid = bullish bias, below = bearish
        htf_1w_bull = close[i] > donch_1w_mid[i]
        htf_1w_bear = close[i] < donch_1w_mid[i]
        
        # === 1d TREND CONFIRMATION ===
        htf_1d_bull = close[i] > donch_1d_mid[i]
        htf_1d_bear = close[i] < donch_1d_mid[i]
        
        # === VOLUME SPIKE FILTER ===
        volume_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        # === RSI CONDITIONS (LOOSE for more trades) ===
        # In uptrend: enter on pullback (RSI < 55)
        # In downtrend: enter on rally (RSI > 45)
        rsi_pullback_long = rsi_14[i] < 55.0
        rsi_pullback_short = rsi_14[i] > 45.0
        rsi_strong_long = rsi_14[i] < 45.0
        rsi_strong_short = rsi_14[i] > 55.0
        
        # === BREAKOUT CONFIRMATION ===
        # Price near upper/lower Donchian confirms trend strength
        breakout_long = close[i] > donch_6h_mid[i]
        breakout_short = close[i] < donch_6h_mid[i]
        
        # === ENTRY LOGIC (LOOSE CONDITIONS FOR TRADE GENERATION) ===
        desired_signal = 0.0
        
        # LONG: 1w bull + 1d bull + RSI pullback + (volume OR breakout)
        if htf_1w_bull and htf_1d_bull:
            if rsi_pullback_long:
                if volume_spike or breakout_long:
                    if rsi_strong_long:
                        # Strong signal
                        desired_signal = SIZE_STRONG
                    else:
                        # Base signal
                        desired_signal = SIZE_BASE
        
        # SHORT: 1w bear + 1d bear + RSI pullback + (volume OR breakout)
        elif htf_1w_bear and htf_1d_bear:
            if rsi_pullback_short:
                if volume_spike or breakout_short:
                    if rsi_strong_short:
                        # Strong signal
                        desired_signal = -SIZE_STRONG
                    else:
                        # Base signal
                        desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (3.0x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 3.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 3.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 3.0 * entry_atr
                else:
                    stop_price = entry_price + 3.0 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals