#!/usr/bin/env python3
"""
Experiment #007: 6h ATR Volatility Breakout + Bollinger Mean Reversion

Hypothesis: Pure mean reversion fails because Bollinger touches often CONTINUE 
trending. Solution: require BOTH volatility expansion (ATR ratio > 1.0) AND 
volume confirmation. This ensures "squeeze has resolveD" before entry.

Key design:
1. ATR(14) / ATR_SMA(20) > 1.0: Confirms volatility expansion = squeeze resolveD
2. Bollinger(20,2) lower touch: Mean reversion entry point
3. Volume > Volume_SMA(20): Confirms institutional interest
4. 1d SMA(20): Trend bias filter (only long above, short below)
5. RSI(14) confirmation: Momentum filter

This is DIFFERENT from previous failures:
- Not pure Donchian (fails consistently)
- Not pure EMA cross (fails consistently)  
- Not Camarilla (overtrades)
- IS ATR + Bollinger + Volume + 1d filter (proven components)

Why it should work in BOTH bull AND bear:
- Bull: Mean reversion to Bollinger lower = buying dips in uptrend
- Bear: ATR expansion often marks capitulation = reversal points
- 1d SMA filter prevents fighting the trend in strong moves

Target: Sharpe > 0.5, 75-150 total train trades, trades >= 10 test
Timeframe: 6h
Size: 0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_atr_volatility_bollinger_1d_v1"
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


def calculate_atr_sma(atr, period=20):
    """SMA of ATR for ratio calculation"""
    return pd.Series(atr).rolling(window=period, min_periods=period).mean().values


def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, sma, lower


def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain = np.insert(gain, 0, 0)
    loss = np.insert(loss, 0, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    mask = avg_loss != 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    
    return rsi


def calculate_volume_sma(volume, period=20):
    """SMA of volume"""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values


def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d SMA for trend filter
    sma_1d_raw = pd.Series(df_1d['close'].values).rolling(window=20, min_periods=20).mean().values
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d_raw)
    
    # Calculate 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_sma_20 = calculate_atr_sma(atr_14, period=20)
    bb_upper, bb_mid, bb_lower = calculate_bollinger(close, period=20, std_mult=2.0)
    rsi_14 = calculate_rsi(close, period=14)
    volume_sma_20 = calculate_volume_sma(volume, period=20)
    
    # Pre-compute ATR ratio
    atr_ratio = np.full(n, np.nan, dtype=np.float64)
    valid_atr = ~np.isnan(atr_14) & ~np.isnan(atr_sma_20) & (atr_sma_20 > 1e-10)
    atr_ratio[valid_atr] = atr_14[valid_atr] / atr_sma_20[valid_atr]
    
    # Pre-compute volume ratio
    vol_ratio = np.full(n, np.nan, dtype=np.float64)
    valid_vol = ~np.isnan(volume) & ~np.isnan(volume_sma_20) & (volume_sma_20 > 1e-10)
    vol_ratio[valid_vol] = volume[valid_vol] / volume_sma_20[valid_vol]
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup period for indicators
    min_bars = 60
    
    for i in range(min_bars, n):
        # Check indicator readiness
        if (np.isnan(atr_14[i]) or atr_14[i] <= 1e-10 or
            np.isnan(bb_lower[i]) or np.isnan(rsi_14[i]) or
            np.isnan(atr_ratio[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(sma_1d_aligned[i]) or np.isnan(bb_mid[i])):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # Current indicator values
        current_atr_ratio = atr_ratio[i]
        current_vol_ratio = vol_ratio[i]
        rsi_val = rsi_14[i]
        price = close[i]
        bb_low = bb_lower[i]
        bb_up = bb_upper[i]
        bb_width = bb_up - bb_low
        sma_1d = sma_1d_aligned[i]
        
        # === VOLATILITY EXPANSION CONFIRMATION ===
        # ATR ratio > 1.0 means volatility is expanding (squeeze resolving)
        vol_expanding = current_atr_ratio > 1.0
        
        # === VOLUME CONFIRMATION ===
        # Volume > 1.5x average confirms institutional interest
        volume_confirmed = current_vol_ratio > 1.5
        
        # === 1d TREND FILTER ===
        price_above_1d = price > sma_1d
        price_below_1d = price < sma_1d
        
        # === RSI CONFIRMATION ===
        rsi_oversold = rsi_val < 35
        rsi_overbought = rsi_val > 65
        rsi_neutral = 35 <= rsi_val <= 65
        
        # === BOLLINGER TOUCH DETECTION ===
        # Price within 0.5% of Bollinger lower = touch
        bb_touch_lower = price <= bb_low * 1.005
        bb_touch_upper = price >= bb_up * 0.995
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG ENTRY: Mean reversion setup
        # 1. Volatility expanding (ATR ratio > 1.0)
        # 2. Volume confirmed (> 1.5x average)
        # 3. Price at/near Bollinger lower (mean reversion entry)
        # 4. RSI oversold (< 35)
        # 5. Price above 1d SMA (trend bias bullish)
        if (vol_expanding and volume_confirmed and 
            bb_touch_lower and rsi_oversold and price_above_1d):
            desired_signal = SIZE
        
        # SHORT ENTRY: Mean reversion setup
        # 1. Volatility expanding
        # 2. Volume confirmed
        # 3. Price at/near Bollinger upper
        # 4. RSI overbought (> 65)
        # 5. Price below 1d SMA (trend bias bearish)
        elif (vol_expanding and volume_confirmed and 
              bb_touch_upper and rsi_overbought and price_below_1d):
            desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TAKE PROFIT (4x ATR - asymmetric, let winners run) ===
        takeprofit_triggered = False
        
        if in_position and position_side > 0:
            profit_target = entry_price + 4.0 * entry_atr
            if high[i] >= profit_target:
                takeprofit_triggered = True
        
        if in_position and position_side < 0:
            profit_target = entry_price - 4.0 * entry_atr
            if low[i] <= profit_target:
                takeprofit_triggered = True
        
        if takeprofit_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE * 0.9:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.9:
            final_signal = -SIZE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                # New position or reversal
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
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