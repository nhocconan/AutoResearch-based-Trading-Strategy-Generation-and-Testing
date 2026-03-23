#!/usr/bin/env python3
"""
Experiment #137: 1d Primary + 1w HTF — KAMA Trend + Donchian Breakout + Choppiness Filter

Hypothesis: Recent failures (#125-136) show complex regime detection creates lag and reduces
trade frequency. This strategy simplifies to proven components:

1) KAMA(21) on 1d — adapts to volatility, reduces whipsaws in chop vs EMA/HMA
2) 1w HMA(21) — macro trend bias only (simple direction filter)
3) Choppiness Index(14) — regime filter: CHOP < 38.2 = trending (trade breakouts)
4) Donchian(20) breakout — entry signal in trend direction
5) ATR(14) 2.5x trailing stop — let winners run, cut losers fast
6) Simple exit: opposite Donchian break OR stoploss hit

Why this should work:
- KAMA adapts speed based on volatility (fast in trends, slow in ranges)
- Choppiness filter prevents breakout trades in ranging markets (major failure mode)
- 1d timeframe naturally produces 25-40 trades/year (low fee drag)
- Simpler exit logic = fewer premature exits
- 1w HMA filter prevents counter-trend trades in bear markets (2022 crash protection)

Position size: 0.30 (discrete, minimizes fee churn)
Stoploss: 2.5*ATR trailing
Target: 25-40 trades/year, Sharpe > 0.5 on ALL symbols (BTC/ETH/SOL)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_donchian_chop_1w_v1"
timeframe = "1d"
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

def calculate_kama(close, period=21, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average.
    KAMA adapts to market noise - fast in trends, slow in ranges.
    """
    close_s = pd.Series(close)
    
    # Efficiency Ratio (ER)
    change = np.abs(close - np.roll(close, period))
    change[0:period] = np.nan
    volatility = pd.Series(np.abs(close - np.roll(close, 1))).rolling(window=period, min_periods=period).sum().values
    er = change / (volatility + 1e-10)
    er[0:period] = np.nan
    
    # Smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(len(close))
    kama[period-1] = close[period-1]
    for i in range(period, len(close)):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (20-day high/low)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2.0
    return upper, lower, mid

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index.
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
    """
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    atr = calculate_atr(high, low, close, period)
    
    # CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    range_val = highest_high - lowest_low
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100.0 * np.log10(atr_sum / (range_val + 1e-10)) / np.log10(period)
    
    chop[np.isnan(chop)] = 50.0
    chop = np.clip(chop, 0, 100)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA for macro trend
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators (ALL before loop - Rule 8)
    atr_14 = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    kama_21 = calculate_kama(close, period=21)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(hma_1w_aligned[i]) or np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(kama_21[i]) or np.isnan(chop_14[i]):
            continue
        
        # === HTF TREND BIAS (1w HMA) ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === 1d TREND FILTER (KAMA slope) ===
        kama_slope = kama_21[i] - kama_21[i-1] if i > 0 else 0.0
        kama_bullish = kama_slope > 0
        kama_bearish = kama_slope < 0
        
        # === CHOPPINESS REGIME FILTER ===
        # CHOP < 38.2 = trending (trade breakouts)
        # CHOP > 61.8 = ranging (skip breakout trades)
        trending_regime = chop_14[i] < 38.2
        ranging_regime = chop_14[i] > 61.8
        
        # === DONCHIAN BREAKOUT ===
        prev_upper = donchian_upper[i-1] if i > 0 else donchian_upper[i]
        prev_lower = donchian_lower[i-1] if i > 0 else donchian_lower[i]
        
        breakout_long = close[i] > prev_upper
        breakout_short = close[i] < prev_lower
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY ---
        # Require: 1w trend up + 1d KAMA up + trending regime + Donchian breakout
        if price_above_hma_1w and kama_bullish and trending_regime and breakout_long:
            new_signal = POSITION_SIZE
        
        # --- SHORT ENTRY ---
        # Require: 1w trend down + 1d KAMA down + trending regime + Donchian breakout
        if price_below_hma_1w and kama_bearish and trending_regime and breakout_short:
            new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        # Hold if still in position and no exit signal
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if price above Donchian mid and 1w trend intact
                if close[i] > donchian_mid[i] and price_above_hma_1w:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if price below Donchian mid and 1w trend intact
                if close[i] < donchian_mid[i] and price_below_hma_1w:
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
        
        # === EXIT ON TREND REVERSAL ===
        if in_position and position_side > 0:
            # Exit long if 1w trend reverses down
            if price_below_hma_1w:
                new_signal = 0.0
            # Exit on opposite Donchian break
            if breakout_short:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 1w trend reverses up
            if price_above_hma_1w:
                new_signal = 0.0
            # Exit on opposite Donchian break
            if breakout_long:
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