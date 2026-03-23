#!/usr/bin/env python3
"""
Experiment #006: 12h Primary + 1d HTF — Regime-Adaptive Donchian + Vol Filter

Hypothesis: After 5 failed strategies using CHOP/CRSI/Fisher combinations,
I'm trying a fundamentally different approach: REGIME-ADAPTIVE with Donchian breakouts.

Why this might work when others failed:
1. Donchian breakouts capture momentum differently than EMA/HMA crossovers (all 5 failed used those)
2. Regime-switching: mean-revert in chop (ADX<20), trend-follow in trends (ADX>25)
3. Volatility filter prevents entries during low-vol traps (ATR ratio > 1.3)
4. 1d HMA provides major trend bias without overfiltering
5. Asymmetric entries: only long when 1d HMA bullish, only short when bearish

Key differences from failed strategies:
- NOT using CHOP, CRSI, Fisher, or KAMA (all tried and failed)
- Using Donchian channel breakouts (20-period high/low)
- Volume confirmation on breakouts (not just price)
- 12h timeframe per experiment spec (not 1h/4h like failed attempts)

Position sizing: 0.28 discrete (conservative for 12h per Rule 4)
Target: 25-45 trades/year on 12h (within Rule 10 limits)
Stoploss: 2.5*ATR trailing stop
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_regime_hma1d_v1"
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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean()
    
    plus_di = 100.0 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / (atr + 1e-10))
    minus_di = 100.0 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / (atr + 1e-10))
    
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA)."""
    close_s = pd.Series(close)
    
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max().values
    lower = low_s.rolling(window=period, min_periods=period).min().values
    
    # Channel width for volatility context
    width = upper - lower
    
    return upper, lower, width

def calculate_volatility_ratio(atr_short, atr_long):
    """Calculate volatility ratio (short-term ATR / long-term ATR)."""
    ratio = atr_short / (atr_long + 1e-10)
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for major trend direction
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 1d ADX for regime filter
    adx_1d = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    
    donchian_upper, donchian_lower, donchian_width = calculate_donchian(high, low, period=20)
    
    # Volatility ratio (short-term vs long-term)
    vol_ratio = calculate_volatility_ratio(atr_7, atr_30)
    
    # Volume SMA for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Price position in Donchian channel
    donchian_pct = (close - donchian_lower) / (donchian_width + 1e-10)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    # ADX hysteresis tracking
    prev_adx_regime = 0  # 0=neutral, 1=trend, 2=range
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(adx_1d_aligned[i]) or np.isnan(donchian_upper[i]):
            continue
        if np.isnan(vol_ratio[i]) or np.isnan(vol_sma[i]):
            continue
        if atr_14[i] == 0 or vol_sma[i] == 0:
            continue
        
        # === 1D TREND BIAS ===
        hma_1d_slope_bull = hma_1d_aligned[i] > hma_1d_aligned[i-5] if i >= 5 else False
        hma_1d_slope_bear = hma_1d_aligned[i] < hma_1d_aligned[i-5] if i >= 5 else False
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === 1D ADX REGIME (with hysteresis) ===
        adx_val = adx_1d_aligned[i]
        
        # Hysteresis: trend regime needs ADX>25 to enter, <18 to exit
        if adx_val > 25.0:
            adx_regime = 1  # Trending
        elif adx_val < 18.0:
            adx_regime = 2  # Range
        else:
            adx_regime = prev_adx_regime  # Keep previous regime
        
        prev_adx_regime = adx_regime
        is_trend_regime = (adx_regime == 1)
        is_range_regime = (adx_regime == 2)
        
        # === VOLATILITY FILTER ===
        # Vol spike: ratio > 1.8 (high vol breakout more reliable)
        # Vol compression: ratio < 1.2 (breakout from squeeze)
        vol_spike = vol_ratio[i] > 1.8
        vol_compression = vol_ratio[i] < 1.2
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 1.0 * vol_sma[i]  # Above average volume
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        # Breakout above upper channel
        donchian_breakout_up = close[i] > donchian_upper[i-1] if i > 0 else False
        # Breakout below lower channel
        donchian_breakout_down = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # Near upper/lower channel (for mean reversion)
        near_upper = donchian_pct[i] > 0.85
        near_lower = donchian_pct[i] < 0.15
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY ---
        # Regime 1: Trending (ADX > 25) + 1d HMA bullish + Donchian breakout up
        if is_trend_regime:
            if hma_1d_slope_bull and price_above_hma_1d:
                if donchian_breakout_up and volume_confirmed:
                    new_signal = POSITION_SIZE
        
        # Regime 2: Range (ADX < 18) + Mean reversion at lower channel
        elif is_range_regime:
            if near_lower and vol_compression:
                if not hma_1d_slope_bear:  # Not strongly bearish on 1d
                    new_signal = POSITION_SIZE
        
        # Vol spike breakout (works in any regime with strong confirmation)
        if vol_spike and donchian_breakout_up and volume_confirmed:
            if price_above_hma_1d or not hma_1d_slope_bear:
                new_signal = POSITION_SIZE
        
        # --- SHORT ENTRY ---
        # Regime 1: Trending (ADX > 25) + 1d HMA bearish + Donchian breakout down
        if is_trend_regime:
            if hma_1d_slope_bear and price_below_hma_1d:
                if donchian_breakout_down and volume_confirmed:
                    new_signal = -POSITION_SIZE
        
        # Regime 2: Range (ADX < 18) + Mean reversion at upper channel
        elif is_range_regime:
            if near_upper and vol_compression:
                if not hma_1d_slope_bull:  # Not strongly bullish on 1d
                    new_signal = -POSITION_SIZE
        
        # Vol spike breakout (works in any regime with strong confirmation)
        if vol_spike and donchian_breakout_down and volume_confirmed:
            if price_below_hma_1d or not hma_1d_slope_bull:
                new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
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
        
        # === EXIT ON TREND FLIP ===
        if in_position and position_side > 0:
            if hma_1d_slope_bear and price_below_hma_1d and is_trend_regime:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if hma_1d_slope_bull and price_above_hma_1d and is_trend_regime:
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