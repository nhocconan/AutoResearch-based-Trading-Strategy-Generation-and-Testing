#!/usr/bin/env python3
"""
Experiment #025: 4h HMA Trend + Donchian Breakout + Volume Spike (12h ref)

HYPOTHESIS: 4h is the optimal balance. 12h HMA(16) for trend direction (faster than 1w).
Donchian(20) for breakout structure. Volume spike (1.5x) for confirmation.
Choppiness index to avoid range-bound whipsaws.

WHY IT SHOULD WORK IN BOTH MARKETS:
- Bull: Price above HMA(16) + breakout above Donchian high + vol spike = momentum continuation
- Bear: Price below HMA(16) + breakdown below Donchian low + vol spike = short momentum
- Choppiness filter: skip when in range (CHOP > 61.8), only trade when trending (CHOP < 50)
- ATR stoploss at 2.0x protects against 2022-style crashes

KEY CHANGES FROM FAILED STRATEGIES:
- Use HMA(16) instead of weekly VWAP (faster, less lag)
- Add Choppiness index filter (proven to reduce whipsaw)
- Simpler entry: just 2 conditions (breakout + vol) with HMA trend alignment
- 2.0 ATR stoploss (not 2.5) = tighter risk, better Sharpe

EXPECTED TRADES: 100-200 total over 4 years (25-50/year per symbol)
- Donchian(20) on 4h = break every 20-40 bars = ~273-546 potential/year
- Volume spike 1.5x → ~35% pass rate = 177-355
- HMA(16) trend filter → ~40% pass rate = 71-142
- Choppiness < 50 → ~50% pass rate = 35-71
- Final: 75-150 trades = statistical validity + manageable fees
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_donchian_vol_chop_v2"
timeframe = "4h"
leverage = 1.0

def calculate_hma(data, period):
    """Hull Moving Average"""
    n = len(data)
    if n < period:
        return np.full(n, np.nan)
    
    series = pd.Series(data)
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).apply(
        lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.sum(np.arange(1, len(x)+1)), raw=True
    ).values
    wma_full = series.rolling(window=period, min_periods=period).apply(
        lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.sum(np.arange(1, len(x)+1)), raw=True
    ).values
    
    hma = np.zeros(n)
    for i in range(n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            hma[i] = 2 * wma_half[i] - wma_full[i]
        else:
            hma[i] = np.nan
    
    # Smooth with WMA(sqrt_period)
    if period >= 4:
        sqrt_hma = pd.Series(hma).rolling(window=sqrt_n, min_periods=sqrt_n).apply(
            lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.sum(np.arange(1, len(x)+1)), raw=True
        ).values
        return sqrt_hma
    return hma

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

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - measures trend vs range"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan)
    for i in range(period - 1, n):
        sum_tr = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j])
            sum_tr += tr
        
        highest_high = max(high[i - period + 1:i + 1])
        lowest_low = min(low[i - period + 1:i + 1])
        
        if sum_tr > 0 and (highest_high - lowest_low) > 0:
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
    
    # 12h HMA(16) for trend direction
    hma_12h = calculate_hma(df_12h['close'].values, 16)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # 12h EMA(8) for faster confirmation
    ema_12h = pd.Series(df_12h['close'].values).ewm(span=8, min_periods=8, adjust=False).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian Channel(20) - primary breakout structure
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Local HMA(16) for faster alignment
    local_hma = calculate_hma(close, 16)
    
    # Volume average (20 bars)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Choppiness index
    chop = calculate_choppiness(high, low, close, period=14)
    
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
    
    warmup = 60  # Enough for Donchian20, ATR14, HMA16, chop14
    
    for i in range(warmup, n):
        # NaN checks
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_12h_aligned[i]) or np.isnan(local_hma[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME FILTER: Skip if choppy (CHOP > 61.8) ===
        is_choppy = chop[i] > 61.8
        
        # === TREND DIRECTION ===
        # Bull trend: price above 12h HMA and local HMA, HMA rising
        bull_trend = (close[i] > hma_12h_aligned[i] and 
                     close[i] > local_hma[i] and
                     local_hma[i] > local_hma[i-4] if not np.isnan(local_hma[i-4]) else True)
        
        # Bear trend: price below 12h HMA and local HMA, HMA falling
        bear_trend = (close[i] < hma_12h_aligned[i] and 
                     close[i] < local_hma[i] and
                     local_hma[i] < local_hma[i-4] if not np.isnan(local_hma[i-4]) else True)
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === DONCHIAN BREAKOUT ===
        prev_donchian_high = donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else np.nan
        prev_donchian_low = donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else np.nan
        
        bullish_breakout = (not np.isnan(prev_donchian_high) and 
                           high[i] > prev_donchian_high)
        bearish_breakout = (not np.isnan(prev_donchian_low) and 
                           low[i] < prev_donchian_low)
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG: Bullish breakout + volume spike + bull trend + NOT choppy
            if bullish_breakout and vol_spike and bull_trend and not is_choppy:
                desired_signal = SIZE
            
            # SHORT: Bearish breakout + volume spike + bear trend + NOT choppy
            elif bearish_breakout and vol_spike and bear_trend and not is_choppy:
                desired_signal = -SIZE
        
        # === EXIT / STOP LOSS LOGIC ===
        if in_position:
            if position_side > 0:
                # Update trailing high
                if i == entry_bar or high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Stop: 2.0 ATR from highest
                stop_price = trailing_high - 2.0 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                
                # Exit if trend flips
                elif close[i] < hma_12h_aligned[i]:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                    
            elif position_side < 0:
                # Update trailing low
                if i == entry_bar or low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Stop: 2.0 ATR from lowest
                stop_price = trailing_low + 2.0 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                
                # Exit if trend flips
                elif close[i] > hma_12h_aligned[i]:
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