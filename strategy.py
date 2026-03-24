#!/usr/bin/env python3
"""
Experiment #155: 6h Primary + 12h/1d HTF — Fisher Transform Reversals + HMA Trend

Hypothesis: 6h timeframe is underexplored (0 experiments) and Fisher Transform excels at
catching reversals in bear/range markets (2022 crash, 2025 bear). Unlike RSI which can
stay extreme for long periods, Fisher Transform normalizes price and highlights turning
points more precisely.

Key insights from 140+ failed experiments:
- Simple EMA/HMA crossover fails on BTC/ETH (always negative Sharpe)
- cRSI didn't work on 6h (#143)
- Strategies with Sharpe=0.000 had ZERO trades (entry conditions too strict)
- #147 worked with loose RSI thresholds and HTF alignment

New approach for 6h:
- 6h Fisher Transform(9) for reversal signals (long when crosses above -1.5, short below +1.5)
- 12h HMA(21) + 1d HMA(50) for major trend bias (both must align)
- 6h HMA(21) for immediate trend confirmation
- ADX(14) > 20 to ensure some trend presence (avoid dead chop)
- ATR(14) 2.5x trailing stop for risk management
- Position size: 0.25 (25% of capital - conservative for reversals)
- Multiple entry tiers to ensure trade generation

Target: 30-60 trades/year, Sharpe>0.167, DD>-40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_hma_adx_12h1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength"""
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, np.nan)
    
    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    # Smoothed DM and TR
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # DI and DX
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    
    for i in range(period, n):
        if tr_s[i] > 1e-10:
            plus_di[i] = 100.0 * plus_dm_s[i] / tr_s[i]
            minus_di[i] = 100.0 * minus_dm_s[i] / tr_s[i]
            
            di_sum = plus_di[i] + minus_di[i]
            if di_sum > 1e-10:
                dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    # ADX = smoothed DX
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - catches reversals in bear rallies
    Normalizes price to highlight turning points
    Long: Fisher crosses above -1.5 (oversold reversal)
    Short: Fisher crosses below +1.5 (overbought reversal)
    """
    n = len(close)
    fisher = np.zeros(n)
    fisher[:] = np.nan
    
    prev_fisher = 0.0
    
    for i in range(period, n):
        # Find highest high and lowest low over period
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        # Normalize price to 0-1 range (with bounds to avoid division issues)
        range_hl = highest - lowest
        if range_hl < 1e-10:
            norm_price = 0.5
        else:
            norm_price = (close[i] - lowest) / range_hl
        
        # Clamp to avoid extremes (0.001 to 0.999)
        norm_price = max(0.001, min(0.999, norm_price))
        
        # Fisher transform: 0.5 * ln((1+x)/(1-x))
        fisher_raw = 0.5 * np.log((1.0 + norm_price) / (1.0 - norm_price))
        
        # Smooth with 1-period lag (standard Fisher implementation)
        fisher[i] = 0.67 * fisher_raw + 0.33 * prev_fisher
        prev_fisher = fisher[i]
    
    return fisher

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA for major trend bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (6h) indicators
    hma_6h = calculate_hma(close, period=21)
    fisher = calculate_fisher_transform(high, low, close, period=9)
    atr = calculate_atr(high, low, close, period=14)
    adx = calculate_adx(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size (conservative for reversals)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(fisher[i]) or np.isnan(hma_6h[i]) or np.isnan(hma_12h_aligned[i]) or 
            np.isnan(hma_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF TREND BIAS (12h + 1d) ===
        # Both HTF must align for strong bias
        htf_bull = (close[i] > hma_12h_aligned[i]) and (close[i] > hma_1d_aligned[i])
        htf_bear = (close[i] < hma_12h_aligned[i]) and (close[i] < hma_1d_aligned[i])
        htf_neutral = not htf_bull and not htf_bear
        
        # === 6h HMA TREND ===
        hma_bull = close[i] > hma_6h[i]
        hma_bear = close[i] < hma_6h[i]
        
        # === ADX FILTER (trend strength) ===
        # Only trade when ADX > 20 (some trend present)
        adx_ok = adx[i] > 20.0
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_long_cross = fisher[i] > -1.5 and fisher[i-1] <= -1.5
        fisher_short_cross = fisher[i] < 1.5 and fisher[i-1] >= 1.5
        
        # Also allow entry when Fisher is at extremes (not just cross)
        fisher_oversold = fisher[i] < -1.2
        fisher_overbought = fisher[i] > 1.2
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # PRIMARY: Fisher reversal + HTF alignment + ADX confirmation
        if fisher_long_cross and htf_bull and hma_bull and adx_ok:
            desired_signal = SIZE
        
        elif fisher_short_cross and htf_bear and hma_bear and adx_ok:
            desired_signal = -SIZE
        
        # SECONDARY: Fisher extreme + HTF alignment (no ADX requirement)
        elif fisher_oversold and htf_bull and hma_bull:
            desired_signal = SIZE * 0.7
        
        elif fisher_overbought and htf_bear and hma_bear:
            desired_signal = -SIZE * 0.7
        
        # TERTIARY: Strong HTF alignment alone (fallback to ensure trades)
        elif htf_bull and hma_bull:
            desired_signal = SIZE * 0.5
        
        elif htf_bear and hma_bear:
            desired_signal = -SIZE * 0.5
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
        elif desired_signal >= SIZE * 0.5:
            final_signal = SIZE * 0.7
        elif desired_signal <= -SIZE * 0.5:
            final_signal = -SIZE * 0.7
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals