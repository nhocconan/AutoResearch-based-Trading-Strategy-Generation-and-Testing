#!/usr/bin/env python3
"""
Experiment #395: 12h Multi-Timeframe Trend with Regime-Adaptive Entries

Hypothesis: After 394 failed experiments, the key insight is that 12h timeframe
needs a BALANCED approach - not too strict (0 trades) and not too loose (whipsaw).

STRATEGY COMPONENTS:
1. 1w HMA(34): Major trend bias - determines long/short bias only
2. 1d HMA(21): Intermediate trend confirmation - must align with 1w
3. 12h Donchian(20): Breakout entry signal - price breaks 20-bar high/low
4. 12h ADX(14): Trend strength filter - ADX > 20 confirms real trend
5. 12h RSI(14): Entry timing - avoid extremes (RSI 35-65 sweet spot)
6. 12h Choppiness(14): Regime filter - CHOP < 55 = trending, enter; CHOP > 55 = range, reduce size

WHY 12h TIMEFRAME:
- Slower than 4h, fewer false signals
- Captures multi-day swings without noise
- Works well with 1d/1w HTF alignment

POSITION SIZING:
- Base: 0.28 (28% of capital)
- Reduced to 0.15 in choppy regime (CHOP > 55)
- Discrete levels: 0.0, ±0.15, ±0.28
- Stoploss: 2.5 * ATR(14) trailing

EXPECTED TRADES:
- 12h has 2 bars/day = ~730 bars/year
- With filters: ~25-40 trades/year per symbol
- Should easily exceed 10 trades on train, 3 on test

Timeframe: 12h (REQUIRED for this experiment)
HTF: 1d and 1w via mtf_data helper (call ONCE before loop)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_trend_1d_1w_hma_adx_rsi_chop_atr_v1"
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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel upper and lower bands."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = high[i - period + 1:i + 1].max()
        lower[i] = low[i - period + 1:i + 1].min()
    
    return upper, lower

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    # Calculate True Range and Directional Movement
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    dm_plus = np.zeros(n)
    dm_minus = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i - 1]
        low_diff = low[i - 1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            dm_plus[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            dm_minus[i] = low_diff
    
    # Smooth using Wilder's method (EMA with span=period)
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=period, min_periods=period, adjust=False).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate DI+ and DI-
    di_plus = np.zeros(n)
    di_minus = np.zeros(n)
    
    for i in range(period, n):
        if tr_smooth[i] > 1e-10:
            di_plus[i] = 100 * dm_plus_smooth[i] / tr_smooth[i]
            di_minus[i] = 100 * dm_minus_smooth[i] / tr_smooth[i]
    
    # Calculate DX and ADX
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = di_plus[i] + di_minus[i]
        if di_sum > 1e-10:
            dx[i] = 100 * np.abs(di_plus[i] - di_minus[i]) / di_sum
    
    # ADX is smoothed DX
    adx_series = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean()
    adx[:] = adx_series.values
    
    return adx

def calculate_rsi(close, period=14):
    """Calculate RSI (Relative Strength Index)."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    for i in range(period, n):
        atr_sum = tr[i - period + 1:i + 1].sum()
        highest_high = high[i - period + 1:i + 1].max()
        lowest_low = low[i - period + 1:i + 1].min()
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1w = calculate_hma(df_1w['close'].values, 34)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    adx = calculate_adx(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    chop = calculate_choppiness_index(high, low, close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_TREND = 0.28  # Full size in strong trend
    SIZE_CHOP = 0.15   # Reduced size in choppy market
    
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
        
        if np.isnan(adx[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        # === 1w HMA MAJOR TREND BIAS ===
        bull_major = close[i] > hma_1w_aligned[i]
        bear_major = close[i] < hma_1w_aligned[i]
        
        # === 1d HMA INTERMEDIATE TREND ===
        bull_intermediate = close[i] > hma_1d_aligned[i]
        bear_intermediate = close[i] < hma_1d_aligned[i]
        
        # === TREND STRENGTH (ADX) ===
        strong_trend = adx[i] > 20
        
        # === REGIME (CHOPPYNESS) ===
        trending_regime = chop[i] < 55
        choppy_regime = chop[i] >= 55
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i - 1] if not np.isnan(donchian_upper[i - 1]) else False
        donchian_breakout_short = close[i] < donchian_lower[i - 1] if not np.isnan(donchian_lower[i - 1]) else False
        
        # === RSI ENTRY TIMING (avoid extremes) ===
        rsi_ok_long = 35 < rsi[i] < 70
        rsi_ok_short = 30 < rsi[i] < 65
        
        # === DETERMINE POSITION SIZE BASED ON REGIME ===
        current_size = SIZE_CHOP if choppy_regime else SIZE_TREND
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # LONG ENTRY: Major bull + Intermediate bull + Strong trend + Breakout + RSI ok
        if bull_major and bull_intermediate and strong_trend:
            if donchian_breakout_long and rsi_ok_long:
                new_signal = current_size
        
        # SHORT ENTRY: Major bear + Intermediate bear + Strong trend + Breakout + RSI ok
        if bear_major and bear_intermediate and strong_trend:
            if donchian_breakout_short and rsi_ok_short:
                new_signal = -current_size
        
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
        if in_position and new_signal != 0.0:
            # Long position should exit if major trend turns bear
            if position_side > 0 and bear_major:
                new_signal = 0.0
            # Short position should exit if major trend turns bull
            if position_side < 0 and bull_major:
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