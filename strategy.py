#!/usr/bin/env python3
"""
Experiment #271: 6h Primary + 1w/1d HTF — Market Structure + MFI Volume Trend v1

Hypothesis: 6h market structure (HH/HL vs LH/LL patterns) combined with volume-weighted 
momentum (MFI) and multi-timeframe trend filters can capture sustained trends while 
avoiding false breakouts. Market structure is lag-free compared to oscillators.

Key innovations vs previous 6h attempts:
1. MARKET STRUCTURE: Pure price action - detect swing highs/lows (5-bar left/right)
   - HH/HL sequence = bullish structure → only long entries
   - LH/LL sequence = bearish structure → only short entries
   - No indicator lag, responds immediately to price changes

2. MFI (Money Flow Index): Volume-weighted RSI equivalent
   - MFI > 50 = buying pressure, MFI < 50 = selling pressure
   - More reliable than RSI alone because it includes volume

3. HTF BIAS: 1w HMA(21) for major trend, 1d HMA(50) for intermediate
   - Only long if price > 1w HMA (major bullish bias)
   - Only short if price < 1w HMA (major bearish bias)
   - 1d HMA confirms intermediate direction

4. VOLUME CONFIRMATION: Volume > SMA(volume, 20) on entry bars
   - Ensures institutional participation in moves

5. ATR STOPLOSS: 2.5x trailing stop to protect capital

Target: 30-60 trades/year on 6h, Sharpe > 0.40 (beat current 0.399)
Position sizing: 0.25 base, 0.30 strong signals (discrete levels)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_market_structure_mfi_vol_1w1d_v1"
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

def calculate_mfi(high, low, close, volume, period=14):
    """
    Money Flow Index - volume-weighted RSI
    Formula: 100 - 100 / (1 + Money Flow Ratio)
    Money Flow Ratio = (Positive Money Flow over n) / (Negative Money Flow over n)
    Typical Price = (High + Low + Close) / 3
    Raw Money Flow = Typical Price * Volume
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Typical Price
    typical_price = (high + low + close) / 3.0
    
    # Raw Money Flow
    raw_money_flow = typical_price * volume
    
    # Positive and Negative Money Flow
    positive_flow = np.zeros(n)
    negative_flow = np.zeros(n)
    
    for i in range(1, n):
        if typical_price[i] > typical_price[i-1]:
            positive_flow[i] = raw_money_flow[i]
            negative_flow[i] = 0.0
        elif typical_price[i] < typical_price[i-1]:
            positive_flow[i] = 0.0
            negative_flow[i] = raw_money_flow[i]
        else:
            positive_flow[i] = 0.0
            negative_flow[i] = 0.0
    
    # Money Flow Ratio
    mfi = np.zeros(n)
    mfi[:] = np.nan
    
    for i in range(period, n):
        pos_sum = np.sum(positive_flow[i-period+1:i+1])
        neg_sum = np.sum(negative_flow[i-period+1:i+1])
        
        if neg_sum < 1e-10:
            mfi[i] = 100.0
        else:
            money_flow_ratio = pos_sum / neg_sum
            mfi[i] = 100.0 - (100.0 / (1.0 + money_flow_ratio))
    
    return mfi

def calculate_sma(values, period):
    """Simple Moving Average"""
    n = len(values)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(values).rolling(window=period, min_periods=period).mean().values
    return sma

def detect_market_structure(high, low, close, swing_bars=5):
    """
    Detect market structure: HH/HL (bullish) vs LH/LL (bearish)
    
    Swing High: bar with higher high than swing_bars bars on each side
    Swing Low: bar with lower low than swing_bars bars on each side
    
    Returns:
    - structure: 1 = bullish (HH/HL), -1 = bearish (LH/LL), 0 = unclear
    - last_swing_high: price of most recent swing high
    - last_swing_low: price of most recent swing low
    """
    n = len(close)
    structure = np.zeros(n)
    swing_highs = np.full(n, np.nan)
    swing_lows = np.full(n, np.nan)
    
    # Detect swing points
    for i in range(swing_bars, n - swing_bars):
        # Swing High
        is_swing_high = True
        for j in range(1, swing_bars + 1):
            if high[i] <= high[i-j] or high[i] <= high[i+j]:
                is_swing_high = False
                break
        if is_swing_high:
            swing_highs[i] = high[i]
        
        # Swing Low
        is_swing_low = True
        for j in range(1, swing_bars + 1):
            if low[i] >= low[i-j] or low[i] >= low[i+j]:
                is_swing_low = False
                break
        if is_swing_low:
            swing_lows[i] = low[i]
    
    # Track structure evolution
    last_sh = np.nan
    last_sl = np.nan
    prev_sh = np.nan
    prev_sl = np.nan
    
    for i in range(swing_bars * 2, n):
        # Update swing points
        if not np.isnan(swing_highs[i]):
            prev_sh = last_sh
            last_sh = swing_highs[i]
        
        if not np.isnan(swing_lows[i]):
            prev_sl = last_sl
            last_sl = swing_lows[i]
        
        # Determine structure
        if not np.isnan(last_sh) and not np.isnan(prev_sh) and not np.isnan(last_sl) and not np.isnan(prev_sl):
            if last_sh > prev_sh and last_sl > prev_sl:
                structure[i] = 1  # Bullish HH/HL
            elif last_sh < prev_sh and last_sl < prev_sl:
                structure[i] = -1  # Bearish LH/LL
            else:
                # Keep previous structure if unclear
                structure[i] = structure[i-1] if i > 0 else 0
        else:
            structure[i] = structure[i-1] if i > 0 else 0
    
    return structure, swing_highs, swing_lows

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (6h) indicators
    atr = calculate_atr(high, low, close, period=14)
    mfi = calculate_mfi(high, low, close, volume, period=14)
    vol_sma = calculate_sma(volume, 20)
    
    # Market structure detection
    structure, swing_highs, swing_lows = detect_market_structure(high, low, close, swing_bars=5)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(mfi[i]) or np.isnan(vol_sma[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(structure[i]) or structure[i] == 0:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === HTF BIAS (1w Major Trend) ===
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === 1d Intermediate Trend ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === MARKET STRUCTURE ===
        bullish_structure = structure[i] == 1  # HH/HL
        bearish_structure = structure[i] == -1  # LH/LL
        
        # === MFI MOMENTUM ===
        mfi_bull = mfi[i] > 50.0
        mfi_bear = mfi[i] < 50.0
        mfi_strong_bull = mfi[i] > 60.0
        mfi_strong_bear = mfi[i] < 40.0
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > vol_sma[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: Bullish structure + MFI bull + 1w HMA bull + volume confirmation
        if bullish_structure and htf_1w_bull and mfi_bull:
            if volume_confirmed and htf_1d_bull:
                # Strong signal: all conditions aligned
                if mfi_strong_bull:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
            elif mfi_strong_bull:
                # Medium signal: strong MFI but no volume spike
                desired_signal = SIZE_BASE
        
        # SHORT: Bearish structure + MFI bear + 1w HMA bear + volume confirmation
        elif bearish_structure and htf_1w_bear and mfi_bear:
            if volume_confirmed and htf_1d_bear:
                if mfi_strong_bear:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
            elif mfi_strong_bear:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
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
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i] if position_side > 0 else 0.0
                lowest_since_entry = low[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i] if position_side > 0 else 0.0
                lowest_since_entry = low[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, low[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals