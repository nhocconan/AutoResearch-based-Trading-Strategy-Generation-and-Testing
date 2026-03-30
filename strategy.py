#!/usr/bin/env python3
"""
Experiment #008: 12h Donchian Breakout + Weekly Trend + Volume + Choppiness

HYPOTHESIS: 12h timeframe naturally produces fewer trades (target 75-150/symbol over 4 years).
Combine proven elements from DB winners:
1. Donchian(20) breakout structure (proven on SOLUSDT 4h: Sharpe 1.38)
2. Weekly EMA(21) for HTF trend direction (avoids counter-trend trades)
3. Volume spike 1.8x confirmation (filters false breakouts)
4. Choppiness < 50 regime filter (skip ranging markets - #1 killer)
5. ATR(14) trailing stop 2.5x (handles 2022 crash volatility)

WHY 12h WORKS:
- Fewer signals = less fee drag (0.10% per round trip)
- Weekly HTF provides strong trend bias
- Choppiness filter avoids whipsaws in 2022-2023 range
- ATR stop scales with volatility (critical for BTC 77% crash)

ENTRY CONDITIONS (ALL must agree - tight filtering):
- Long: Price > Donchian_upper_prev + Close > Weekly_EMA21 + Vol > 1.8x + CHOP < 50
- Short: Price < Donchian_lower_prev + Close < Weekly_EMA21 + Vol > 1.8x + CHOP < 50

EXIT CONDITIONS:
- ATR trailing stop: 2.5x from entry/high (long) or entry/low (short)
- HTF trend flip: Weekly EMA crosses against position
- Choppiness spike: CHOP > 61.8 (market became ranging)

TARGET: 75-150 trades per symbol over 4 years (19-37/year)
SIZE: 0.28 (28% position - discrete level to minimize churn)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_weekly_trend_vol_chop_1w_v1"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = ranging (SKIP entries)
    CHOP < 50 = trending (GOOD for entries)
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA(21) for strong trend bias
    ema_21_1w = pd.Series(df_1w['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    donchian_up, donchian_lo = calculate_donchian(high, low, period=20)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Volume ratio (20-period MA) - 1.8x threshold
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size - discrete level
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    trailing_high = 0.0
    trailing_low = 0.0
    
    warmup = 250  # 200 for Donchian + 14 for CHOP + 20 for vol MA + HTF alignment
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(ema_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_up[i]) or np.isnan(donchian_lo[i]):
            signals[i] = 0.0
            continue
        
        # === CHOPPINESS REGIME FILTER ===
        chop_value = chop[i]
        is_choppy = chop_value > 61.8  # SKIP if ranging
        is_trending = chop_value < 50   # ENTER if trending
        
        # === HTF TREND: Weekly EMA(21) direction ===
        htf_trend_up = close[i] > ema_aligned[i]
        htf_trend_down = close[i] < ema_aligned[i]
        
        # === VOLUME CONFIRMATION (1.8x) ===
        vol_spike = vol_ratio[i] > 1.8
        
        # === DONCHIAN BREAKOUT ===
        # Use PREVIOUS bar's Donchian levels to avoid look-ahead
        prev_donchian_up = donchian_up[i - 1]
        prev_donchian_lo = donchian_lo[i - 1]
        
        breakout_up = close[i] > prev_donchian_up
        breakout_down = close[i] < prev_donchian_lo
        
        # === ENTRY LOGIC (ALL conditions must agree) ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Trending + breakout up + HTF trend up + volume spike ===
            if breakout_up and htf_trend_up and vol_spike and is_trending:
                desired_signal = SIZE
            
            # === SHORT: Trending + breakout down + HTF trend down + volume spike ===
            if breakout_down and htf_trend_down and vol_spike and is_trending:
                desired_signal = -SIZE
        
        # === EXIT/STOPLOSS LOGIC ===
        if in_position:
            if position_side > 0:
                # Update trailing high for long position
                if i == entry_bar or high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Trailing stop: exit if price falls 2.5 ATR from recent high
                stop_price = trailing_high - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Exit if HTF trend flips against position
                if htf_trend_down:
                    desired_signal = 0.0
                
                # Exit if market becomes choppy (ranging)
                if is_choppy:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Update trailing low for short position
                if i == entry_bar or low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Trailing stop: exit if price rises 2.5 ATR from recent low
                stop_price = trailing_low + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Exit if HTF trend flips against position
                if htf_trend_up:
                    desired_signal = 0.0
                
                # Exit if market becomes choppy (ranging)
                if is_choppy:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 3 bars to avoid fee churn on 12h ===
        if in_position and (i - entry_bar) < 3:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION STATE ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
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