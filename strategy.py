#!/usr/bin/env python3
"""
Experiment #560: 6h Primary + 1d/1w HTF — Fisher Transform + Donchian Breakout + Volume

Hypothesis: 6h timeframe with Ehlers Fisher Transform provides superior reversal detection
vs RSI during bear/range markets (2022 crash, 2025 bear). Fisher normalizes price to Gaussian
distribution, creating clear overbought/oversold levels (-2 to +2). Combined with Donchian
breakout confirmation and volume filter, this catches trend reversals early while avoiding
false breakouts.

Key differences from failed 6h experiments:
1. Fisher Transform instead of RSI - better reversal detection in bear markets
2. Donchian(20) breakout confirmation - ensures momentum behind entry
3. Volume ratio filter (>1.2x avg) - filters false breakouts
4. Dual HTF: 1d HMA for medium bias + 1w HMA for macro bias
5. Simpler entry logic - Fisher cross + Donchian break + volume confirm
6. ATR-based stoploss (2.5x) on all positions

Strategy logic:
1. 1w HMA(21) = macro trend bias (very slow filter)
2. 1d HMA(21) = medium trend bias
3. 6h Fisher(9) = reversal detection (cross above -1.5 = long, cross below +1.5 = short)
4. 6h Donchian(20) = breakout confirmation (price breaks 20-bar high/low)
5. 6h Volume Ratio = volume confirmation (vol > 1.2x 20-bar avg)
6. 6h ATR(14)*2.5 stoploss on all positions

Entry conditions (all must align):
- LONG: Fisher crosses above -1.5 + price breaks Donchian high + volume > 1.2x avg + HTF bull
- SHORT: Fisher crosses below +1.5 + price breaks Donchian low + volume > 1.2x avg + HTF bear

Target: Sharpe>0.40, trades>=30 train (7.5/year), trades>=3 test
Timeframe: 6h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_donchian_vol_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Creates clear overbought/oversold levels (-2 to +2)
    Better reversal detection than RSI in bear/range markets
    
    Formula:
    1. Calculate typical price: (high + low) / 2
    2. Normalize to -1 to +1 range using period high/low
    3. Apply Fisher transform: 0.5 * ln((1+x)/(1-x))
    4. Smooth with EMA
    """
    n = len(close)
    if n < period + 5:
        return np.full(n, np.nan)
    
    # Typical price
    typical = (high + low) / 2.0
    
    # Normalize to -1 to +1 using rolling high/low
    fisher_raw = np.zeros(n)
    fisher_raw[:] = np.nan
    
    for i in range(period, n):
        highest = np.nanmax(high[i-period+1:i+1])
        lowest = np.nanmin(low[i-period+1:i+1])
        price_range = highest - lowest
        
        if price_range > 1e-10:
            # Normalize typical price to -0.99 to +0.99 (avoid division by zero)
            normalized = 0.99 * (2.0 * (typical[i] - lowest) / price_range - 1.0)
            normalized = np.clip(normalized, -0.99, 0.99)
            
            # Fisher transform
            fisher_raw[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
        else:
            fisher_raw[i] = 0.0
    
    # Smooth with EMA
    fisher = pd.Series(fisher_raw).ewm(span=3, min_periods=3, adjust=False).mean().values
    
    return fisher

def calculate_donchian(high, low, period=20):
    """
    Donchian Channel - tracks highest high and lowest low over period
    Returns: upper_channel, lower_channel, middle_channel
    """
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period - 1, n):
        upper[i] = np.nanmax(high[i-period+1:i+1])
        lower[i] = np.nanmin(low[i-period+1:i+1])
    
    middle = (upper + lower) / 2.0
    
    return upper, lower, middle

def calculate_volume_ratio(volume, period=20):
    """
    Volume Ratio - current volume vs rolling average
    Ratio > 1.2 = above average volume (confirmation)
    """
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / vol_avg
    
    vol_ratio[:period] = np.nan
    
    return vol_ratio

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
    fisher = calculate_fisher_transform(high, low, close, period=9)
    donchian_upper, donchian_lower, donchian_middle = calculate_donchian(high, low, period=20)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    atr = calculate_atr(high, low, close, period=14)
    
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
        
        if np.isnan(fisher[i]) or np.isnan(donchian_upper[i]) or np.isnan(vol_ratio[i]):
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
        
        # === HTF BIAS (1w macro + 1d medium) ===
        htf_bull = close[i] > hma_1d_aligned[i] and close[i] > hma_1w_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i] and close[i] < hma_1w_aligned[i]
        htf_neutral = not htf_bull and not htf_bear
        
        # === FISHER TRANSFORM SIGNALS ===
        # Fisher crosses above -1.5 = bullish reversal
        # Fisher crosses below +1.5 = bearish reversal
        fisher_bull_cross = False
        fisher_bear_cross = False
        
        if i >= 2 and not np.isnan(fisher[i-1]) and not np.isnan(fisher[i-2]):
            # Bullish: Fisher was below -1.5, now crosses above
            if fisher[i-1] < -1.5 and fisher[i] >= -1.5:
                fisher_bull_cross = True
            # Bearish: Fisher was above +1.5, now crosses below
            if fisher[i-1] > 1.5 and fisher[i] <= 1.5:
                fisher_bear_cross = True
        
        # Also check for extreme levels (stronger signal)
        fisher_extreme_bull = fisher[i] < -1.8
        fisher_extreme_bear = fisher[i] > 1.8
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_high = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        donchian_breakout_low = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === VOLUME CONFIRMATION ===
        volume_confirm = vol_ratio[i] > 1.2 if not np.isnan(vol_ratio[i]) else False
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG ENTRY: Fisher bull cross + Donchian breakout + volume + HTF bull
        if fisher_bull_cross and donchian_breakout_high and volume_confirm:
            if htf_bull:
                desired_signal = SIZE_STRONG
            elif htf_neutral:
                desired_signal = SIZE_BASE
            # Even in bear HTF, allow smaller position if Fisher extreme
            elif fisher_extreme_bull:
                desired_signal = SIZE_BASE * 0.6
        
        # SHORT ENTRY: Fisher bear cross + Donchian breakout + volume + HTF bear
        if fisher_bear_cross and donchian_breakout_low and volume_confirm:
            if htf_bear:
                desired_signal = -SIZE_STRONG
            elif htf_neutral:
                desired_signal = -SIZE_BASE
            # Even in bull HTF, allow smaller position if Fisher extreme
            elif fisher_extreme_bear:
                desired_signal = -SIZE_BASE * 0.6
        
        # Alternative: Fisher extreme without full breakout (catch early reversals)
        if desired_signal == 0.0:
            if fisher_extreme_bull and htf_bull and volume_confirm:
                desired_signal = SIZE_BASE * 0.8
            elif fisher_extreme_bear and htf_bear and volume_confirm:
                desired_signal = -SIZE_BASE * 0.8
        
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