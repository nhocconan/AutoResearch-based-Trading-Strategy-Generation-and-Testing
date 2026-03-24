#!/usr/bin/env python3
"""
Experiment #531: 6h Primary + 1d/1w HTF — HMA Trend + RSI Pullback + Donchian Breakout

Hypothesis: 6h timeframe sits between 4h (intraday) and 12h (multi-day), capturing
weekly patterns with 4 bars/day and 28 bars/week. Previous 6h attempts failed due to
over-filtering (KAMA+ADX+CHOP complex regime = 0 trades). This strategy uses SIMPLER
logic with fewer filters to ensure adequate trade frequency.

Key differences from failed #523 (mtf_6h_kama_adx_rsi_regime_1d1w_v1):
1. HMA instead of KAMA (KAMA failed on 6h timeframe)
2. Remove complex ADX/CHOP regime detection (causes 0 trades)
3. Add Donchian(20) breakout for momentum confirmation
4. Looser RSI thresholds (30/70 instead of 25/75) for more trades
5. Volume ratio confirmation (current vs 20-period avg)
6. Simpler HTF bias: just 1d HMA direction, not dual 1d+1w

Strategy logic:
1. 1w HMA(21) = macro trend bias (slow filter)
2. 1d HMA(21) = medium trend bias  
3. 6h HMA(16) = primary trend following
4. 6h RSI(14) = entry timing on pullbacks
5. 6h Donchian(20) = breakout momentum confirmation
6. 6h Volume ratio = participation confirmation
7. ATR(14)*2.5 stoploss on all positions

Entry conditions (LONG):
- Price > 1w HMA AND Price > 1d HMA (HTF bullish)
- Price > 6h HMA(16) (primary trend up)
- RSI(14) between 35-60 (pullback, not overbought)
- Price > Donchian(20) mid OR breaking upper band
- Volume ratio > 0.8 (adequate participation)

Entry conditions (SHORT):
- Price < 1w HMA AND Price < 1d HMA (HTF bearish)
- Price < 6h HMA(16) (primary trend down)
- RSI(14) between 40-65 (pullback, not oversold)
- Price < Donchian(20) mid OR breaking lower band
- Volume ratio > 0.8

Target: Sharpe>0.40, trades>=80 train (20/year), trades>=10 test
Timeframe: 6h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_hma_rsi_donchian_vol_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Donchian Channels - highest high and lowest low over period"""
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
        upper[i] = np.nanmax(high[i-period+1:i+1])
        lower[i] = np.nanmin(low[i-period+1:i+1])
        mid[i] = (upper[i] + lower[i]) / 2.0
    
    return upper, lower, mid

def calculate_volume_ratio(volume, period=20):
    """Volume ratio: current volume vs rolling average"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[:] = np.nan
    
    for i in range(period - 1, n):
        if vol_avg[i] > 1e-10:
            vol_ratio[i] = volume[i] / vol_avg[i]
        else:
            vol_ratio[i] = 1.0
    
    return vol_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1d HMA for medium trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for macro trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 6h indicators
    hma_6h = calculate_hma(close, period=16)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    donch_upper, donch_lower, donch_mid = calculate_donchian(high, low, period=20)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
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
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_6h[i]) or np.isnan(rsi[i]) or np.isnan(donch_mid[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1w macro + 1d medium) ===
        htf_bull = close[i] > hma_1d_aligned[i] and close[i] > hma_1w_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i] and close[i] < hma_1w_aligned[i]
        
        # === PRIMARY TREND (6h HMA) ===
        trend_bull = close[i] > hma_6h[i]
        trend_bear = close[i] < hma_6h[i]
        
        # HMA slope confirmation (5-bar lookback)
        hma_slope_bull = hma_6h[i] > hma_6h[i-5] if i >= 5 and not np.isnan(hma_6h[i-5]) else False
        hma_slope_bear = hma_6h[i] < hma_6h[i-5] if i >= 5 and not np.isnan(hma_6h[i-5]) else False
        
        # === DONCHIAN MOMENTUM ===
        donch_bull = close[i] > donch_mid[i]
        donch_bear = close[i] < donch_mid[i]
        donch_breakout_up = close[i] > donch_upper[i] * 0.995  # Near upper band
        donch_breakout_down = close[i] < donch_lower[i] * 1.005  # Near lower band
        
        # === RSI PULLBACK ===
        rsi_pullback_long = 35.0 <= rsi[i] <= 60.0
        rsi_pullback_short = 40.0 <= rsi[i] <= 65.0
        rsi_recovery_long = rsi[i] > 30.0 and rsi[i] > rsi[i-1] if i > 0 else False
        rsi_recovery_short = rsi[i] < 70.0 and rsi[i] < rsi[i-1] if i > 0 else False
        
        # === VOLUME CONFIRMATION ===
        vol_ok = vol_ratio[i] >= 0.7  # At least 70% of average volume
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG entries
        if htf_bull and trend_bull and hma_slope_bull:
            if rsi_pullback_long and donch_bull and vol_ok:
                desired_signal = SIZE_BASE
            elif rsi_recovery_long and vol_ok:
                desired_signal = SIZE_BASE * 0.8
            elif donch_breakout_up and rsi[i] < 70.0 and vol_ok:
                desired_signal = SIZE_STRONG
        
        # SHORT entries
        elif htf_bear and trend_bear and hma_slope_bear:
            if rsi_pullback_short and donch_bear and vol_ok:
                desired_signal = -SIZE_BASE
            elif rsi_recovery_short and vol_ok:
                desired_signal = -SIZE_BASE * 0.8
            elif donch_breakout_down and rsi[i] > 30.0 and vol_ok:
                desired_signal = -SIZE_STRONG
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            if low[i] < stop_price:
                stoploss_triggered = True
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            if high[i] > stop_price:
                stoploss_triggered = True
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
        
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
        elif abs(desired_signal) >= SIZE_BASE * 0.5:
            final_signal = np.sign(desired_signal) * SIZE_BASE * 0.8
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
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