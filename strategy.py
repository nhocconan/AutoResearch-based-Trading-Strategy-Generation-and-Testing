#!/usr/bin/env python3
"""
Experiment #164: 4h Primary + 12h/1d HTF — Multi-HMA Trend Following with ADX Filter

Hypothesis: Previous CRSI mean-reversion strategies failed because 2025 bear/range market
punishes counter-trend entries. This strategy switches to PURE trend-following:

1) 4h HMA(21) - primary trend direction
2) 12h HMA(48) - intermediate trend confirmation (slower than 4h)
3) 1d HMA(21) - macro bias (only trade WITH 1d trend)
4) ADX(14) > 20 - filter out choppy markets (no trades when ADX < 20)
5) Donchian(20) breakout - entry trigger in trend direction
6) ATR(14) trailing stop - 2.5x ATR for risk management
7) Volume filter - volume > 0.8x 20-bar avg

Key differences from failed strategies:
- NO mean reversion (CRSI) - pure trend following
- ADX filter prevents trades in chop (major cause of whipsaw losses)
- All 3 HTF HMAs must align for full position size
- Donchian breakout provides clear entry timing (not RSI extremes)

Target: 20-50 trades/year on 4h, Sharpe > 0.5 on ALL symbols
Position size: 0.25 base, 0.30 with full HTF confluence
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_adx_donchian_12h1d_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_adx(high, low, close, period=14):
    """
    Calculate ADX (Average Directional Index).
    ADX > 25 = trending, ADX < 20 = ranging/choppy
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.clip(lower=0)
    minus_dm = minus_dm.clip(lower=0)
    
    # Filter: +DM only if > -DM, -DM only if > +DM
    plus_dm = plus_dm.where(plus_dm > minus_dm, 0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0)
    
    # Smoothed values (Wilder's smoothing = EMA with span=period)
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100.0 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / (atr + 1e-10))
    minus_di = 100.0 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / (atr + 1e-10))
    
    # DX and ADX
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values, plus_di.values, minus_di.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    adx_14, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    hma_4h = calculate_hma(close, period=21)
    
    # Calculate 12h HMA for intermediate trend
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=48)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate 1d HMA for macro bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Volume average (20-bar)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.25
    POSITION_SIZE_MAX = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_4h[i]) or np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(adx_14[i]) or np.isnan(donchian_upper[i]):
            continue
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(vol_avg[i]) or vol_avg[i] == 0:
            continue
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * vol_avg[i]
        
        # === ADX TREND STRENGTH FILTER ===
        # Only trade when ADX > 20 (trending market)
        trend_strength_ok = adx_14[i] > 20.0
        
        # === HTF TREND ALIGNMENT ===
        # 4h trend
        price_above_hma_4h = close[i] > hma_4h[i]
        price_below_hma_4h = close[i] < hma_4h[i]
        
        # 12h trend
        price_above_hma_12h = close[i] > hma_12h_aligned[i]
        price_below_hma_12h = close[i] < hma_12h_aligned[i]
        
        # 1d macro bias
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY ---
        # All HTF trends bullish + ADX strong + Donchian breakout + volume
        if (price_above_hma_4h and price_above_hma_12h and price_above_hma_1d and
            trend_strength_ok and breakout_long and volume_ok):
            # Full size if all 3 HTF aligned
            new_signal = POSITION_SIZE_MAX
        
        # Partial long if 4h + 12h aligned (1d neutral or bullish)
        elif (price_above_hma_4h and price_above_hma_12h and
              trend_strength_ok and breakout_long and volume_ok):
            new_signal = POSITION_SIZE_BASE
        
        # --- SHORT ENTRY ---
        # All HTF trends bearish + ADX strong + Donchian breakout + volume
        if (price_below_hma_4h and price_below_hma_12h and price_below_hma_1d and
            trend_strength_ok and breakout_short and volume_ok):
            # Full size if all 3 HTF aligned
            new_signal = -POSITION_SIZE_MAX
        
        # Partial short if 4h + 12h aligned (1d neutral or bearish)
        elif (price_below_hma_4h and price_below_hma_12h and
              trend_strength_ok and breakout_short and volume_ok):
            new_signal = -POSITION_SIZE_BASE
        
        # === HOLD POSITION LOGIC ===
        # Hold if in position and trend still valid (no exit signal yet)
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if price still above 4h HMA and ADX still strong
                if price_above_hma_4h and adx_14[i] > 18.0:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if price still below 4h HMA and ADX still strong
                if price_below_hma_4h and adx_14[i] > 18.0:
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
        
        # === TREND REVERSAL EXIT ===
        # Exit long if price crosses below 4h HMA
        if in_position and position_side > 0 and price_below_hma_4h:
            new_signal = 0.0
        
        # Exit short if price crosses above 4h HMA
        if in_position and position_side < 0 and price_above_hma_4h:
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
                # Position flip
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