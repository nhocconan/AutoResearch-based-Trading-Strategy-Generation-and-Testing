#!/usr/bin/env python3
"""
Experiment #025: 4h Donchian Breakout + HMA Trend + Volume + Choppiness Regime

HYPOTHESIS:
- 4h timeframe: proven to work (DB winners all use 4h with 100-300 trades)
- Donchian(20): captures momentum breakouts every 20-40 bars on 4h
- HMA(48): smoother trend direction, avoids short-term noise
- Volume spike: 1.5x average confirms institutional interest
- Choppiness regime: avoid sideways markets (CHOP > 61.8 = no entry)
- HTF 12h: confirms larger trend direction before entry

WHY IT SHOULD WORK IN BOTH MARKETS:
- Bull: Breakout + volume + HMA up + not choppy = ride momentum higher
- Bear: Breakdown + volume + HMA down + not choppy = short the breakdown
- Choppiness filter prevents whipsaw in range-bound periods

EXPECTED TRADES: 75-200 total over 4 years (19-50/year)
- Donchian(20) on 4h = potential break every 20-40 bars = ~219-438/year
- Volume spike 1.5x → reduces by ~40%
- HMA trend filter → reduces by ~30%
- Choppiness filter (>61.8 = no trade) → reduces by ~25%
- Final: ~75-175 trades = statistical validity
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_hma_vol_chop_v3"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(data, period):
    """Hull Moving Average"""
    series = pd.Series(data)
    half_length = period // 2
    sqrt_length = int(np.sqrt(period))
    
    wma1 = series.rolling(window=half_length, min_periods=half_length).mean()
    wma2 = series.rolling(window=period, min_periods=period).mean()
    diff = 2 * wma1 - wma2
    
    hma = diff.rolling(window=sqrt_length, min_periods=sqrt_length).mean()
    return hma.values

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CEI): measures market choppiness vs trending
    CHOP > 61.8 = choppy/range (avoid)
    CHOP < 38.2 = trending (good for momentum)
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    chop = np.zeros(n, dtype=np.float64)
    
    for i in range(period - 1, n):
        sum_tr = 0.0
        for j in range(period):
            tr_val = max(high[i - j] - low[i - j], 
                        abs(high[i - j] - close[i - j - 1]) if i - j - 1 >= 0 else high[i - j] - low[i - j])
            sum_tr += tr_val
        
        highest_high = max(high[i - period + 1:i + 1])
        lowest_low = min(low[i - period + 1:i + 1])
        
        if highest_high - lowest_low > 0:
            chop[i] = 100 * np.log10(sum_tr / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_12h = get_htf_data(prices, '12h')
    
    # === Calculate HTF HMA for trend confirmation ===
    hma_12h = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # === Calculate local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian Channel(20) - captures momentum breakouts
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # HMA(48) for local trend
    hma_48 = calculate_hma(close, 48)
    
    # Choppiness Index(14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Volume average (20 bars)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # === Signals ===
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
    
    warmup = 80  # Enough for Donchian20, HMA48, ATR14, chop14
    
    for i in range(warmup, n):
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME FILTER: Choppiness < 61.8 (not too choppy) ===
        not_choppy = chop[i] < 61.8
        
        # === TREND: HMA(48) direction + HTF 12h HMA confirmation ===
        hma_trending_up = hma_48[i] > hma_48[i-1] if not np.isnan(hma_48[i-1]) else True
        hma_trending_down = hma_48[i] < hma_48[i-1] if not np.isnan(hma_48[i-1]) else False
        
        # HTF trend: 12h HMA above/below price
        htf_trend_up = (not np.isnan(hma_12h_aligned[i]) and 
                        close[i] > hma_12h_aligned[i] and 
                        hma_12h_aligned[i] > hma_12h_aligned[i-1] if not np.isnan(hma_12h_aligned[i-1]) else True)
        htf_trend_down = (not np.isnan(hma_12h_aligned[i]) and 
                          close[i] < hma_12h_aligned[i])
        
        bull_trend = hma_trending_up and htf_trend_up
        bear_trend = hma_trending_down and htf_trend_down
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === DONCHIAN BREAKOUT ===
        prev_donchian_high = donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else np.nan
        prev_donchian_low = donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else np.nan
        
        bullish_breakout = (not np.isnan(prev_donchian_high) and 
                           close[i] > prev_donchian_high)
        bearish_breakout = (not np.isnan(prev_donchian_low) and 
                           close[i] < prev_donchian_low)
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG: Bullish breakout + volume spike + bull trend + not choppy
            if bullish_breakout and vol_spike and bull_trend and not_choppy:
                desired_signal = SIZE
            
            # SHORT: Bearish breakout + volume spike + bear trend + not choppy
            elif bearish_breakout and vol_spike and bear_trend and not_choppy:
                desired_signal = -SIZE
        
        # === EXIT LOGIC ===
        if in_position:
            if position_side > 0:
                # Trailing high
                if i == entry_bar or high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Stop: 2.5 ATR from highest
                stop_price = trailing_high - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                
                # Exit if trend reverses (HMA turns down)
                elif hma_48[i] < hma_48[i-2] if not np.isnan(hma_48[i-2]) else False:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                    
            elif position_side < 0:
                # Trailing low
                if i == entry_bar or low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Stop: 2.5 ATR from lowest
                stop_price = trailing_low + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                
                # Exit if trend reverses (HMA turns up)
                elif hma_48[i] > hma_48[i-2] if not np.isnan(hma_48[i-2]) else False:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
        
        # === MINIMUM HOLD: 4 bars to reduce fee churn ===
        if in_position and (i - entry_bar) < 4:
            desired_signal = position_side * SIZE
        
        # === EXECUTE NEW POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                trailing_high = high[i]
                trailing_low = low[i]
        
        signals[i] = desired_signal
    
    return signals