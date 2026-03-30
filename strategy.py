#!/usr/bin/env python3
"""
Experiment #022: 12h Donchian Breakout + 1d HMA Trend + Choppiness Regime

HYPOTHESIS: Simple 12h breakout strategy with HTF trend confirmation:
1. 12h Donchian(20) breakout - proven price channel structure
2. 1d HMA(21) trend direction - HTF confirmation reduces whipsaws
3. Choppiness Index < 45 - only trade in trending markets
4. Volume spike > 1.5x - breakout validation
5. ATR stoploss (2.5x) - risk management

WHY IT SHOULD WORK IN BOTH BULL AND BEAR:
- Bull: Price breaks 20-period high + HTF uptrend + volume = strong long
- Bear: Price breaks 20-period low + HTF downtrend + volume = strong short
- Range (CHOP > 55): No trades, avoids chop losses

KEY INSIGHT: 12h has proven best Sharpe ratios in DB (1.3-1.5). 
Donchian breakout is the single most reliable pattern across 16K+ experiments.
Simple logic = fewer trades = less fee drag = better generalization.

TARGET: 75-150 total trades over 4 years (19-37/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_hma_chop_vol_v1"
timeframe = "12h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 2:
        return np.full(n, np.nan)
    
    tr = np.maximum(high[1:] - low[1:], 
                    np.maximum(np.abs(high[1:] - close[:-1]), 
                               np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[high[0] - low[0]], tr])
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Donchian Channel - 20 period high/low"""
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_hma(data, period=21):
    """Hull Moving Average - faster response than SMA"""
    n = len(data)
    if n < period:
        return np.full(n, np.nan)
    
    # WMA with period/2
    half_period = max(1, period // 2)
    wma_half = pd.Series(data).rolling(window=half_period, min_periods=half_period).apply(
        lambda x: np.sum(x * np.arange(1, len(x)+1)) / np.sum(np.arange(1, len(x)+1)), raw=True
    ).values
    
    # WMA with period
    wma_full = pd.Series(data).rolling(window=period, min_periods=period).apply(
        lambda x: np.sum(x * np.arange(1, len(x)+1)) / np.sum(np.arange(1, len(x)+1)), raw=True
    ).values
    
    # HMA = 2*WMA(period/2) - WMA(period)
    hma = 2 * wma_half - wma_full
    return hma

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = choppy market (no trend)
    CHOP < 38.2 = trending market (trend following works)
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        # Sum of ATR over period
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], 
                    abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j],
                    abs(low[j] - close[j-1]) if j > 0 else high[j] - low[j])
            atr_sum += tr
        
        # Highest high - lowest low over period
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        range_sum = highest_high - lowest_low
        
        if range_sum > 0:
            # CHOP = 100 * log10(sum(ATR)) / log10(range)
            chop[i] = 100 * np.log10(atr_sum) / np.log10(range_sum)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d HMA for HTF trend direction ===
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    
    # HTF: HMA rising = bull, falling = bear
    htf_hma_diff = np.zeros(len(df_1d))
    for i in range(5, len(df_1d)):
        if not np.isnan(hma_1d[i]) and not np.isnan(hma_1d[i-5]):
            htf_hma_diff[i] = hma_1d[i] - hma_1d[i-5]
    
    htf_bullish = htf_hma_diff > 0
    htf_bearish = htf_hma_diff < 0
    
    # Align HTF to 12h
    htf_bullish_aligned = align_htf_to_ltf(prices, df_1d, htf_bullish.astype(float))
    htf_bearish_aligned = align_htf_to_ltf(prices, df_1d, htf_bearish.astype(float))
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Volume ratio (20-period)
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
    
    warmup = 200  # Donchian needs 20, ATR needs 14, chop needs 14
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # === ENTRY CONDITIONS ===
        desired_signal = 0.0
        
        # Choppiness filter: only trade when CHOP < 45 (trending)
        trending = chop[i] < 45.0
        
        # Volume confirmation
        vol_confirm = vol_ratio[i] > 1.5
        
        # HTF trend
        htf_bull = htf_bullish_aligned[i] > 0.5 if not np.isnan(htf_bullish_aligned[i]) else False
        htf_bear = htf_bearish_aligned[i] > 0.5 if not np.isnan(htf_bearish_aligned[i]) else False
        
        if not in_position:
            # === LONG: Breakout above 20-period high + HTF bull + volume + trending ===
            price_breaks_high = close[i] > donchian_upper[i]
            
            if price_breaks_high and htf_bull and vol_confirm and trending:
                desired_signal = SIZE
            
            # === SHORT: Breakout below 20-period low + HTF bear + volume + trending ===
            price_breaks_low = close[i] < donchian_lower[i]
            
            if price_breaks_low and htf_bear and vol_confirm and trending:
                desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR trailing) ===
        if in_position:
            if position_side > 0:
                # Long stop: entry - 2.5 * ATR
                stop_price = entry_price - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Exit if price falls back below donchian
                if close[i] < donchian_lower[i]:
                    desired_signal = 0.0
                
                # Exit if HTF turns bearish
                if htf_bear:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Short stop: entry + 2.5 * ATR
                stop_price = entry_price + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Exit if price rises back above donchian
                if close[i] > donchian_upper[i]:
                    desired_signal = 0.0
                
                # Exit if HTF turns bullish
                if htf_bull:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 3 bars to avoid fee churn ===
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
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals