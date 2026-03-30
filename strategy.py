#!/usr/bin/env python3
"""
Experiment #005: 12h Donchian + Choppiness + Volume + 1d Trend

HYPOTHESIS: Simple price-channel breakout with regime filter works in both bull and bear:
- 12h Donchian(20) breakout identifies structural breaks
- Choppiness Index < 50 = trending (follow), > 60 = ranging (reduce)
- Volume confirmation validates breakouts
- 1d HTF trend filter prevents counter-trend entries

WHY IT WORKS IN BOTH MARKETS:
- Bull: Donchian breakout + chop<50 + vol spike + 1d bull = ride rallies
- Bear: Donchian breakdown + chop<50 + vol spike + 1d bear = short breakdowns
- Range (2022): chop>60 = no trades, avoids whipsaws at bottom

KEY INSIGHT from DB: "ONE strong signal + volume + regime filter" = 1.49 test Sharpe.
Following proven pattern from mtf_4h_chop_donchian_vol_regime_12h_v1 (best: 1.491).

TARGET: 75-175 total trades over 4 years (19-44/year). HARD MAX: 200.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_chop_vol_1d_v1"
timeframe = "12h"
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - returns upper/lower bands"""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - lower = trending, higher = ranging
    CHOP < 38.2 = strong trend
    CHOP > 61.8 = ranging (mean reversion territory)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        # Sum of true range over period
        tr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = high[j] - low[j]
            if j > 0:
                tr = max(tr, abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            tr_sum += tr
        
        # Highest - lowest over period
        hh = np.max(high[i - period + 1:i + 1])
        ll = np.min(low[i - period + 1:i + 1])
        range_size = hh - ll
        
        if range_size > 0 and tr_sum > 0:
            chop[i] = 100 * np.log10(tr_sum / range_size) / np.log10(period)
    
    return chop

def calculate_sma(data, period):
    """Simple Moving Average"""
    sma = pd.Series(data).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # === 1d HTF trend: SMA(21) direction ===
    sma_21_1d = calculate_sma(close_1d, 21)
    htf_bull = close_1d > sma_21_1d
    htf_bear = close_1d < sma_21_1d
    
    # Align HTF to 12h
    htf_bull_aligned = align_htf_to_ltf(prices, df_1d, htf_bull.astype(float))
    htf_bear_aligned = align_htf_to_ltf(prices, df_1d, htf_bear.astype(float))
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian Channel 20
    donchian_high, donchian_low = calculate_donchian(high, low, period=20)
    
    # Choppiness Index
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Volume ratio (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # === Signals ===
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
    
    warmup = 60  # Donchian needs 20, chop needs 14, vol needs 20
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # === Regime check ===
        # Chop < 50 = trending (good for breakouts)
        # Chop > 60 = ranging (reduce or skip)
        trending = chop[i] < 50.0
        ranging = chop[i] > 60.0
        
        # === Volume confirmation ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === HTF trend ===
        htf_bull_trend = htf_bull_aligned[i] > 0.5 if not np.isnan(htf_bull_aligned[i]) else False
        htf_bear_trend = htf_bear_aligned[i] > 0.5 if not np.isnan(htf_bear_aligned[i]) else False
        
        # === Donchian breakout signals ===
        # Previous bar inside channel, current bar breaks out
        prev_inside = (high[i-1] <= donchian_high[i-1] * 1.001) and (low[i-1] >= donchian_low[i-1] * 0.999)
        bullish_breakout = close[i] > donchian_high[i] and prev_inside
        bearish_breakout = close[i] < donchian_low[i] and prev_inside
        
        # Strong breakout (more lenient) - price at new 20-bar high/low
        strong_bullish = close[i] > donchian_high[i] and close[i-1] <= donchian_high[i-1]
        strong_bearish = close[i] < donchian_low[i] and close[i-1] >= donchian_low[i-1]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG: Bullish breakout + trending + volume + HTF bull
            if (bullish_breakout or strong_bullish) and vol_spike and trending:
                if htf_bull_trend or not htf_bear_trend:  # Neutral or bull
                    desired_signal = SIZE
            
            # SHORT: Bearish breakout + trending + volume + HTF bear
            if (bearish_breakout or strong_bearish) and vol_spike and trending:
                if htf_bear_trend or not htf_bull_trend:  # Neutral or bear
                    desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR trailing stop) ===
        if in_position:
            if position_side > 0:
                # Update trailing high
                if high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Trailing stop: 2.5 ATR from highest point
                stop_price = trailing_high - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Exit if HTF turns bearish
                if htf_bear_trend:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Update trailing low
                if low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Trailing stop: 2.5 ATR from lowest point
                stop_price = trailing_low + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Exit if HTF turns bullish
                if htf_bull_trend:
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
                trailing_high = high[i]
                trailing_low = low[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals