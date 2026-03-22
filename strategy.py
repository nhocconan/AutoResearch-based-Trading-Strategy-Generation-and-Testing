#!/usr/bin/env python3
"""
Experiment #431: 12h Dual HMA Trend + Donchian Breakout + Volume Confirmation

Hypothesis: After analyzing 430+ failed experiments, the pattern is clear:
- Complex regime filters (Choppiness, Fisher, BB Squeeze) FAIL on 12h
- Mean reversion strategies FAIL in 2025 bear market
- 12h needs SIMPLE trend-following with STRONG HTF confirmation

This strategy uses:
1. 1d HMA(21) for primary trend bias (long only above, short only below)
2. 1w HMA(21) for major trend confirmation (avoids counter-trend in major bear)
3. 12h Donchian(15) breakout for entry (faster than Donchian(20) for more trades)
4. Volume confirmation (1.3x 20-bar average) to filter false breakouts
5. ATR(14) 2.5x trailing stop for risk management
6. Position sizing: 0.25 discrete (conservative for 12h volatility)

Why this should work on 12h:
- Fewer whipsaws than 4h (12h closes are more significant)
- Donchian breakout captures sustained trends, not noise
- Dual HMA (1d + 1w) provides strong trend filter
- Volume confirmation reduces false breakouts
- Should generate 20-40 trades/year (meets min 10 trades requirement)

Timeframe: 12h (REQUIRED)
HTF: 1d and 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_dual_hma_donchian_vol_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_donchian(high, low, period=15):
    """Calculate Donchian Channel (highest high, lowest low over period)."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = high[i-period+1:i+1].max()
        lower[i] = low[i-period+1:i+1].min()
    
    return upper, lower

def calculate_volume_ma(volume, period=20):
    """Calculate volume moving average."""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_ma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 15)
    vol_ma = calculate_volume_ma(volume, 20)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(vol_ma[i]) or vol_ma[i] == 0:
            signals[i] = 0.0
            continue
        
        # === 1d HMA TREND BIAS ===
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === 1w HMA MAJOR TREND CONFIRMATION ===
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        # Break above previous bar's Donchian upper
        donchian_long = close[i] > donchian_upper[i-1]
        # Break below previous bar's Donchian lower
        donchian_short = close[i] < donchian_lower[i-1]
        
        # === VOLUME CONFIRMATION ===
        vol_confirm = volume[i] > 1.3 * vol_ma[i]
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # LONG: 1d bullish + (1w bullish OR neutral) + Donchian breakout + Volume
        if bull_trend_1d and donchian_long and vol_confirm:
            # Extra confirmation: 1w should not be strongly bearish
            if bull_trend_1w or (hma_1w_aligned[i] > 0):
                new_signal = SIZE
        
        # SHORT: 1d bearish + (1w bearish OR neutral) + Donchian breakout + Volume
        elif bear_trend_1d and donchian_short and vol_confirm:
            # Extra confirmation: 1w should not be strongly bullish
            if bear_trend_1w or (hma_1w_aligned[i] < 0):
                new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit long if 1d trend turns bearish
        if in_position and position_side > 0 and bear_trend_1d:
            new_signal = 0.0
        
        # Exit short if 1d trend turns bullish
        if in_position and position_side < 0 and bull_trend_1d:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals