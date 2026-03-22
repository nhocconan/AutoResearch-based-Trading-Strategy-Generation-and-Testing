#!/usr/bin/env python3
"""
Experiment #529: 4h Primary + 1d HTF — KAMA Trend + ADX + Donchian Breakout

Hypothesis: After 474 failed strategies (mostly volspike/choppiness/connors combos),
try a PROVEN trend-following approach with adaptive trend (KAMA) + trend strength (ADX)
+ breakout confirmation (Donchian). This combination has worked in prior experiments.

Key insights from failures:
- Volatility spike strategies: ALL failed (volspike_* in history)
- Choppiness index: Recent failures (#522, #523, #524)
- Connors RSI: Not working as expected (#525, #528)
- Complex regime switching: Underperforming

Why KAMA + ADX + Donchian might work:
1. KAMA (Kaufman Adaptive MA) adapts to market noise - faster in trends, slower in chop
2. ADX filters out weak trends (ADX > 20 = tradable trend)
3. Donchian breakout confirms momentum (price breaking 20-period high/low)
4. 1d HMA provides major trend direction filter
5. Simple logic = consistent signals across BTC/ETH/SOL

Position sizing: 0.25-0.30 (discrete levels)
Stoploss: 2.5 * ATR trailing stop
Target: 25-45 trades/symbol on train, >=3 on test, Sharpe > 0 all symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_adx_donchian_1d_v1"
timeframe = "4h"
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

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market noise: fast in trends, slow in chop.
    Formula: KAMA = KAMA_prev + SC * (Price - KAMA_prev)
    SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
    ER = Change / Noise (Efficiency Ratio)
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    close_s = pd.Series(close)
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(close - np.roll(close, period))
    change[:period] = np.nan
    
    noise = np.zeros(n)
    for i in range(period, n):
        noise[i] = np.sum(np.abs(close[i-period+1:i+1] - np.roll(close[i-period+1:i+1], 1)))
    
    noise[:period] = np.nan
    er = change / (noise + 1e-10)
    er = np.clip(er, 0, 1)
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Calculate KAMA
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA with SMA
    kama[period-1] = np.nanmean(close[:period])
    
    for i in range(period, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_adx(high, low, close, period=14):
    """
    Calculate Average Directional Index (ADX).
    Measures trend strength (not direction).
    ADX > 20 = trending market, ADX < 20 = ranging market.
    """
    n = len(close)
    
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    # Smoothed values using Wilder's method (EMA with span=period)
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / (atr + 1e-10))
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / (atr + 1e-10))
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def calculate_donchian(high, low, period=20):
    """
    Calculate Donchian Channel (upper and lower bands).
    Upper = highest high over period
    Lower = lowest low over period
    Breakout above upper = bullish signal
    Breakout below lower = bearish signal
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max().values
    lower = low_s.rolling(window=period, min_periods=period).min().values
    
    return upper, lower

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA) - reduces lag vs EMA."""
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        return series.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    hma_raw = 2.0 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_n)
    
    return hma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF HMA for major trend direction
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_50 = calculate_hma(df_1d['close'].values, period=50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    
    # KAMA for adaptive trend (period=10, fast=2, slow=30)
    kama_4h_10 = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    
    # ADX for trend strength
    adx_14 = calculate_adx(high, low, close, 14)
    
    # Donchian channels for breakout confirmation
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_50_aligned[i]):
            continue
        if np.isnan(kama_4h_10[i]):
            continue
        if np.isnan(adx_14[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === 1D MAJOR TREND (primary direction filter) ===
        bull_regime = close[i] > hma_1d_21_aligned[i]
        bear_regime = close[i] < hma_1d_21_aligned[i]
        
        # 1d HMA slope for trend strength confirmation
        hma_slope_bull = hma_1d_21_aligned[i] > hma_1d_50_aligned[i]
        hma_slope_bear = hma_1d_21_aligned[i] < hma_1d_50_aligned[i]
        
        # === 4H KAMA TREND ===
        kama_bull = close[i] > kama_4h_10[i]
        kama_bear = close[i] < kama_4h_10[i]
        
        # KAMA slope (current vs 5 bars ago)
        kama_slope_bull = kama_4h_10[i] > kama_4h_10[i-5] if i >= 5 else False
        kama_slope_bear = kama_4h_10[i] < kama_4h_10[i-5] if i >= 5 else False
        
        # === ADX TREND STRENGTH ===
        trend_strong = adx_14[i] > 20.0  # Trending market
        trend_very_strong = adx_14[i] > 25.0
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_up = close[i] > donchian_upper[i-1]  # Break above previous upper
        donchian_breakout_down = close[i] < donchian_lower[i-1]  # Break below previous lower
        
        # === ENTRY LOGIC — TREND + BREAKOUT CONFLUENCE ===
        new_signal = 0.0
        
        # LONG ENTRIES
        # Condition 1: Bull regime + KAMA bull + ADX strong + Donchian breakout
        if bull_regime and kama_bull and trend_strong and donchian_breakout_up:
            new_signal = POSITION_SIZE
        # Condition 2: Bull regime + KAMA slope up + ADX very strong (momentum entry)
        elif bull_regime and kama_slope_bull and trend_very_strong:
            new_signal = POSITION_SIZE * 0.9
        # Condition 3: Strong bull (1d HMA slope) + KAMA bull + Donchian breakout
        elif bull_regime and hma_slope_bull and kama_bull and donchian_breakout_up:
            new_signal = POSITION_SIZE
        
        # SHORT ENTRIES (mirror logic)
        if new_signal == 0.0:
            # Condition 1: Bear regime + KAMA bear + ADX strong + Donchian breakout
            if bear_regime and kama_bear and trend_strong and donchian_breakout_down:
                new_signal = -POSITION_SIZE
            # Condition 2: Bear regime + KAMA slope down + ADX very strong
            elif bear_regime and kama_slope_bear and trend_very_strong:
                new_signal = -POSITION_SIZE * 0.9
            # Condition 3: Strong bear (1d HMA slope) + KAMA bear + Donchian breakout
            elif bear_regime and hma_slope_bear and kama_bear and donchian_breakout_down:
                new_signal = -POSITION_SIZE
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT CONDITIONS (regime flip or trend weakness) ===
        # Exit long on regime flip to bear or ADX dropping below 18
        if in_position and position_side > 0:
            if bear_regime and hma_slope_bear:
                new_signal = 0.0
            elif adx_14[i] < 18.0:  # Trend weakening
                new_signal = 0.0
        
        # Exit short on regime flip to bull or ADX dropping below 18
        if in_position and position_side < 0:
            if bull_regime and hma_slope_bull:
                new_signal = 0.0
            elif adx_14[i] < 18.0:  # Trend weakening
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Flip position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals