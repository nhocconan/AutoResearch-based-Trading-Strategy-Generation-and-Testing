#!/usr/bin/env python3
"""
Experiment #021: 4h Camarilla Pivot Breakout + Volume Spike + Choppiness Regime

HYPOTHESIS: Combine proven DB elements (Camarilla pivot + volume + chop regime):
1. Camarilla pivot levels (R3/R4 breakout = strong momentum, S3/S4 = bear continuation)
2. Volume spike 2.0x confirmation - filters fake breakouts
3. Choppiness < 50 (trending) to avoid range markets
4. ATR-based stoploss for risk management

WHY IT SHOULD WORK:
- Camarilla pivots are tighter than traditional pivots - more trade signals
- R4/S4 breakouts are rare but high-probability momentum moves
- Volume confirms breakout legitimacy
- Choppiness filters ranging markets where pivots fail

TARGET: 75-200 total trades over 4 years (similar to winning #008)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_camarilla_pivot_vol_chop_v2"
timeframe = "4h"
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

def calculate_camarilla(high, low, close):
    """
    Camarilla Pivot Levels
    R4 = High + 3 * (Close - Low) / 2
    R3 = High + 2 * (Close - Low) / 2
    R2 = High + (Close - Low) * 1.1 / 2
    R1 = High + (Close - Low) * 0.55 / 2
    S1 = Low - (Close - High) * 0.55 / 2
    S2 = Low - (Close - High) * 1.1 / 2
    S3 = Low - (Close - High) * 2 / 2
    S4 = Low - (Close - High) * 3 / 2
    """
    n = len(close)
    r4 = np.full(n, np.nan)
    r3 = np.full(n, np.nan)
    r3_5 = np.full(n, np.nan)  # Between R3 and R4
    s3_5 = np.full(n, np.nan)  # Between S3 and S4
    s3 = np.full(n, np.nan)
    s4 = np.full(n, np.nan)
    
    for i in range(1, n):
        h, l, c = high[i], low[i], close[i]
        range_hl = h - l
        range_hc = h - c
        range_ch = c - l
        
        if range_hl > 0:
            r4[i] = h + 3 * range_ch / 2
            r3[i] = h + 2 * range_ch / 2
            r3_5[i] = h + 2.5 * range_ch / 2
            s3_5[i] = l - 2.5 * range_ch / 2
            s3[i] = l - 2 * range_ch / 2
            s4[i] = l - 3 * range_ch / 2
    
    return r3, r4, s3, s4, r3_5, s3_5

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = ranging - DON'T enter
    CHOP < 50 = trending - GOOD to enter
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan)
    for i in range(period, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        
        if highest > lowest and atr_sum > 0:
            range_hl = highest - lowest
            chop[i] = 100 * np.log10(atr_sum / range_hl) / np.log10(period)
    
    return chop

def calculate_hma(close, period=21):
    """Hull Moving Average for trend"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    close_s = pd.Series(close)
    wma_half = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean().values
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean().values
    
    hull = 2 * wma_half - wma_full
    hma = pd.Series(hull).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean().values
    return hma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_12h = get_htf_data(prices, '12h')
    
    # 12h HMA for trend direction
    hma_21_12h = calculate_hma(df_12h['close'].values, period=21)
    hma_aligned = align_htf_to_ltf(prices, df_12h, hma_21_12h)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    r3, r4, s3, s4, r3_5, s3_5 = calculate_camarilla(high, low, close)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Volume ratio (20-period MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
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
    
    warmup = 250
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === CHOPPINESS REGIME FILTER ===
        chop_value = chop[i]
        is_choppy = chop_value > 61.8
        is_trending = chop_value < 50
        
        # === HTF TREND: 12h HMA(21) direction ===
        htf_trend_up = close[i] > hma_aligned[i]
        htf_trend_down = close[i] < hma_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 2.0
        
        # === CAMARILLA PIVOT BREAKOUT ===
        # Long: price breaks above R4 (extreme momentum) OR R3.5 (strong)
        # Short: price breaks below S4 (extreme momentum) OR S3.5 (strong)
        
        breakout_r4 = close[i] > r4[i-1] if not np.isnan(r4[i-1]) else False
        breakout_r3_5 = close[i] > r3_5[i-1] if not np.isnan(r3_5[i-1]) else False
        breakout_s4 = close[i] < s4[i-1] if not np.isnan(s4[i-1]) else False
        breakout_s3_5 = close[i] < s3_5[i-1] if not np.isnan(s3_5[i-1]) else False
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: R4 breakout (extreme) OR R3.5 breakout + HTF up + volume + trending ===
            if (breakout_r4 or breakout_r3_5) and htf_trend_up and vol_spike and is_trending:
                desired_signal = SIZE
            
            # === SHORT: S4 breakout (extreme) OR S3.5 breakout + HTF down + volume + trending ===
            if (breakout_s4 or breakout_s3_5) and htf_trend_down and vol_spike and is_trending:
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
                
                # Exit if HTF trend flips
                if htf_trend_down:
                    desired_signal = 0.0
                
                # Exit if market becomes choppy
                if is_choppy:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Update trailing low
                if i == entry_bar or low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Trailing stop
                stop_price = trailing_low + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Exit if HTF trend flips
                if htf_trend_up:
                    desired_signal = 0.0
                
                # Exit if market becomes choppy
                if is_choppy:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 4 bars to avoid fee churn ===
        if in_position and (i - entry_bar) < 4:
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