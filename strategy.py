#!/usr/bin/env python3
"""
Experiment #021: 4h Donchian Breakout + 1d HMA + ADX Regime + Simple Volume

HYPOTHESIS: Simplify the current best strategy (#015, Sharpe=0.513, 210 trades)
by replacing Choppiness with ADX regime filter. ADX is simpler to calculate
and more intuitive: ADX > 20 = trending (enter), ADX < 15 = ranging (avoid).

RATIONALE:
- Current #015 uses CHOP<50 (complex, logs/log calculations)
- ADX is simpler: just directional movement + smoothed average
- ADX > 20 reliably identifies trending markets for breakouts
- Keep everything else that works: 12h HMA trend + 4h Donchian + volume

TARGET: 120-180 trades over 4 years (30-45/year)
- Donchian(20) on 4h: ~6 breakouts per month per direction
- With HTF filter + volume: expect ~50% to qualify
- With ADX regime: expect ~60% to qualify
- Net: ~30-45 qualified entries/year = 120-180 over 4 years (in range)

TIMEFRAME: 4h primary
MTF: 12h HMA(21) for trend direction
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_hma_adx_vol_12h_v3"
timeframe = "4h"
leverage = 1.0

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index (ADX)
    ADX > 20 = trending market (good for breakouts)
    ADX < 15 = ranging market (avoid)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    # True Range
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Directional Movement
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    # Smoothed values using Wilder's smoothing
    atr_smooth = pd.Series(tr).ewm(alpha=1/period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/period, min_periods=period, adjust=False).mean().values
    
    # DX calculation
    plus_di = np.zeros(n, dtype=np.float64)
    minus_di = np.zeros(n, dtype=np.float64)
    dx = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        if atr_smooth[i] > 0:
            plus_di[i] = 100 * plus_dm_smooth[i] / atr_smooth[i]
            minus_di[i] = 100 * minus_dm_smooth[i] / atr_smooth[i]
            
            di_sum = plus_di[i] + minus_di[i]
            if di_sum > 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    # ADX = smoothed DX
    adx = pd.Series(dx).ewm(alpha=1/period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di

def calculate_hma(close, period=21):
    """Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    close_s = pd.Series(close)
    wma_half = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean().values
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean().values
    
    hull = 2 * wma_half - wma_full
    hma = pd.Series(hull).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean().values
    return hma

def calculate_donchian(high, low, period=20):
    """Donchian Channel"""
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
    df_12h = get_htf_data(prices, '12h')
    
    # 12h HMA(21) for trend direction
    hma_21_12h = calculate_hma(df_12h['close'].values, period=21)
    hma_aligned = align_htf_to_ltf(prices, df_12h, hma_21_12h)
    
    # === Local 4h indicators ===
    donchian_up, donchian_lo = calculate_donchian(high, low, period=20)
    adx, plus_di = calculate_adx(high, low, close, period=14)
    
    # Volume ratio (20-period MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.28
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    trailing_high = 0.0
    trailing_low = 0.0
    
    warmup = 100  # 20 for donchian + 14 for ADX + 20 for vol MA + HTF alignment buffer
    
    for i in range(warmup, n):
        # Check indicator readiness
        if np.isnan(adx[i]) or np.isnan(hma_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_up[i]) or np.isnan(donchian_lo[i]):
            signals[i] = 0.0
            continue
        
        # === ADX REGIME FILTER ===
        # ADX > 20 = trending (good for breakouts)
        # ADX < 15 = ranging (avoid entries)
        adx_value = adx[i]
        is_trending = adx_value > 20
        is_ranging = adx_value < 15
        
        # === HTF TREND: 12h HMA(21) direction ===
        htf_trend_up = close[i] > hma_aligned[i]
        htf_trend_down = close[i] < hma_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === DONCHIAN BREAKOUT ===
        # Long: price breaks ABOVE previous channel high
        # Short: price breaks BELOW previous channel low
        prev_donchian_up = donchian_up[i - 1]
        prev_donchian_lo = donchian_lo[i - 1]
        
        breakout_up = close[i] > prev_donchian_up
        breakout_down = close[i] < prev_donchian_lo
        
        # === ATR for stoploss calculation ===
        # Simple ATR calculation for trailing stop
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i-1]) if i > 0 else 0
        tr3 = abs(low[i] - close[i-1]) if i > 0 else 0
        atr_current = max(tr1, tr2, tr3)
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Trending + breakout up + HTF trend up + volume spike ===
            if breakout_up and htf_trend_up and vol_spike and is_trending:
                desired_signal = SIZE
            
            # === SHORT: Trending + breakout down + HTF trend down + volume spike ===
            if breakout_down and htf_trend_down and vol_spike and is_trending:
                desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR trailing stop) ===
        if in_position:
            if position_side > 0:
                # Update trailing high
                if i == entry_bar or high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Trailing stop
                stop_price = trailing_high - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Exit if HTF trend flips
                if htf_trend_down:
                    desired_signal = 0.0
                
                # Exit if ranging market
                if is_ranging:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Update trailing low
                if i == entry_bar or low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Trailing stop
                stop_price = trailing_low + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Exit if HTF trend flips
                if htf_trend_up:
                    desired_signal = 0.0
                
                # Exit if ranging market
                if is_ranging:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 4 bars to avoid fee churn ===
        if in_position and (i - entry_bar) < 4:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_current if atr_current > 0 else 1.0
                entry_bar = i
                trailing_high = high[i]
                trailing_low = low[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals