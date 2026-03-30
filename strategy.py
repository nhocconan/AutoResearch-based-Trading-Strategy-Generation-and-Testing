#!/usr/bin/env python3
"""
Experiment #025: 12h Donchian(20) + 1d HMA Trend + Volume + Choppiness Regime

HYPOTHESIS: 12h timeframe is optimal trade frequency (target 19-37/year).
Donchian(20) on 12h gives ~73 potential breakouts/year.
1d HMA provides structural trend (bull/bear/range).
Volume spike (1.5x) + Choppiness Index (<61.8) filters reduce to 19-37 trades.
ATR-based stoploss at 2.5x handles volatility.

WHY IT SHOULD WORK IN BOTH MARKETS:
- Bull: Breakout above 20-bar high + 1d HMA up + volume spike + CHOP<61.8 → long
- Bear: Breakdown below 20-bar low + 1d HMA down + volume spike + CHOP<61.8 → short
- Choppiness Index ensures we only trade trending markets, not range
- 2022 bear: HMA down filter prevents buying, allows shorts
- 2023 recovery: HMA up filter allows longs, stops out quickly if fake

EXPECTED TRADES: 75-150 total over 4 years (19-37/year per symbol)
- Donchian(20) on 12h = 73 potential/year
- Volume spike (1.5x) → ~60% trigger rate = 44/year
- Choppiness (<61.8) → ~50% trigger rate = 22/year
- HTF trend confirmation → ~85% trigger rate = 19-22/year
- Final estimate: 75-100 total over 4 years = statistical validity
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_hma_vol_chop_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(data, period):
    """Hull Moving Average"""
    half = pd.Series(data).rolling(window=period // 2, min_periods=period // 2).mean()
    full = pd.Series(data).rolling(window=period, min_periods=period).mean()
    hma = (2 * half - full)
    hma = hma.rolling(window=int(np.sqrt(period)), min_periods=int(np.sqrt(period))).mean()
    return hma.values

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

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = market is choppy/ranging (don't trade)
    CHOP < 38.2 = market is trending (good to trade)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        highest_high = high[i-period+1:i+1].max()
        lowest_low = low[i-period+1:i+1].min()
        
        if highest_high > lowest_low:
            sum_tr = 0.0
            for j in range(i-period+1, i+1):
                sum_tr += high[j] - low[j]
            
            chop[i] = 100 * np.log10(sum_tr / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d HMA(21) for structural trend
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness_index(high, low, close, period=14)
    
    # Donchian Channel(20)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
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
    
    warmup = 50  # Enough for Donchian20, ATR14, CHOP14
    
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
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === TREND DIRECTION: 1d HMA ===
        prev_hma = hma_1d_aligned[i-1] if i > 0 and not np.isnan(hma_1d_aligned[i-1]) else hma_1d_aligned[i]
        curr_hma = hma_1d_aligned[i]
        
        hma_bullish = close[i] > curr_hma and curr_hma > prev_hma  # HMA rising + price above
        hma_bearish = close[i] < curr_hma and curr_hma < prev_hma  # HMA falling + price below
        
        # === CHOPPINESS REGIME ===
        # CHOP < 61.8 = trending market (good to trade)
        # CHOP > 61.8 = ranging market (avoid - mean reversion only)
        chop_trending = chop[i] < 61.8
        
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
            # LONG: Bullish breakout + volume spike + bull trend + trending market
            if bullish_breakout and vol_spike and hma_bullish and chop_trending:
                desired_signal = SIZE
            
            # SHORT: Bearish breakout + volume spike + bear trend + trending market
            elif bearish_breakout and vol_spike and hma_bearish and chop_trending:
                desired_signal = -SIZE
        
        # === EXIT LOGIC: ATR Trailing Stop ===
        if in_position:
            if position_side > 0:
                # Update trailing high
                if i == entry_bar or high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Stop: 2.5 ATR from highest
                stop_price = trailing_high - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                    trailing_high = 0.0
                
                # Exit if HMA trend flips to bearish
                elif hma_bearish:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                    
            elif position_side < 0:
                # Update trailing low
                if i == entry_bar or low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Stop: 2.5 ATR from lowest
                stop_price = trailing_low + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                    trailing_low = 0.0
                
                # Exit if HMA trend flips to bullish
                elif hma_bullish:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
        
        # === MINIMUM HOLD: 3 bars to reduce fee churn ===
        if in_position and (i - entry_bar) < 3:
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