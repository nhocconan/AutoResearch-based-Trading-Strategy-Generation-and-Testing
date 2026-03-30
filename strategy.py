#!/usr/bin/env python3
"""
Experiment #008: 12h Donchian + Weekly HMA Trend + ADX Regime + Volume

HYPOTHESIS: Multi-timeframe breakout with regime filtering on 12h.
- 12h captures structural moves without overtrading (vs 4h)
- 1w HMA(16) provides macro trend direction (bull/bear filter)
- ADX(14) confirms it's a trending environment (not choppy)
- Donchian(10) is tight enough for ~100-200 trades over 4 years
- Volume spike confirms institutional conviction

WHY IT SHOULD WORK IN BOTH MARKETS:
1. Weekly HMA filters out counter-trend trades in bear markets
2. ADX > 20 ensures only trending environments trigger entries
3. Donchian(10) captures ~3.3-day swings - reasonable for 12h
4. Tight stop at 2*ATR limits losses in whipsaws
5. 12h timeframe = ~73-150 trades/4yr (proven viable range from DB)

EXPECTED TRADE COUNT:
- 12h bars/4yr ≈ 2920
- Donchian(10) breakout: ~1 per 15-25 bars = ~117-195 raw signals
- ADX > 20 filter: ~55% qualify = ~64-107
- Volume spike (1.3x): ~50% pass = ~32-54
- Weekly HMA align: ~75% align = ~24-40 trades/symbol
- SAFE RANGE: 24-40 trades/4yr (may need looser filters)

ENTRY CONDITIONS (3 conditions + volume):
- Weekly HMA(16) aligned with direction (above for longs, below for shorts)
- ADX(14) > 20 (trending, not choppy)
- Donchian(10) breakout (prior bar breaks outside 9-bar range)
- Volume > 1.3x 20-bar MA

EXIT: 2*ATR stoploss OR opposite breakout.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_1w_hma_adx_vol_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(data, period):
    """Hull Moving Average"""
    half = pd.Series(data).rolling(window=period // 2, min_periods=period // 2).mean()
    full = pd.Series(data).rolling(window=period, min_periods=period).mean()
    hma = (2 * half - full)
    hma = hma.rolling(window=int(np.sqrt(period)), min_periods=int(np.sqrt(period))).mean()
    return hma.values

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - vectorized"""
    n = len(close)
    
    # True Range and Directional Movement
    tr = np.zeros(n, dtype=np.float64)
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        # True Range
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Directional Movement
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    # Smooth using Wilder's method
    tr_smooth = pd.Series(tr).ewm(alpha=1/period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/period, min_periods=period, adjust=False).mean().values
    
    # DI calculations
    plus_di = np.zeros(n, dtype=np.float64)
    minus_di = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        if tr_smooth[i] > 1e-10:
            plus_di[i] = 100 * plus_dm_smooth[i] / tr_smooth[i]
            minus_di[i] = 100 * minus_dm_smooth[i] / tr_smooth[i]
    
    # DX
    dx = np.zeros(n, dtype=np.float64)
    for i in range(n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    # ADX
    adx = pd.Series(dx).ewm(alpha=1/period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === Load Weekly data ONCE for trend ===
    df_weekly = get_htf_data(prices, '1w')
    close_weekly = df_weekly['close'].values.astype(np.float64)
    hma_16w = calculate_hma(close_weekly, 16)
    # Align weekly to 12h bars with shift(1) to avoid look-ahead
    hma_16w_aligned = align_htf_to_ltf(prices, df_weekly, hma_16w)
    
    # === Local 12h indicators ===
    adx_14 = calculate_adx(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian Channel(10) - price structure
    donchian_upper = pd.Series(high).rolling(window=10, min_periods=10).max().values
    donchian_lower = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # Volume average for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 1e-10, vol_ma, 1.0)
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    
    warmup = 50  # Enough for Donchian10, ATR14, ADX14, HMA16
    
    for i in range(warmup, n):
        # NaN checks
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(adx_14[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME FILTERS ===
        # Weekly HMA trend (price above = bullish, below = bearish)
        weekly_hma_val = hma_16w_aligned[i]
        if np.isnan(weekly_hma_val):
            signals[i] = 0.0
            continue
        
        weekly_bullish = close[i] > weekly_hma_val
        weekly_bearish = close[i] < weekly_hma_val
        
        # ADX > 20 = trending (not choppy)
        is_trending = adx_14[i] > 20.0
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.3
        
        # === DONCHIAN BREAKOUT (prior bar's range) ===
        prev_upper = donchian_upper[i-1] if i > 0 and not np.isnan(donchian_upper[i-1]) else np.nan
        prev_lower = donchian_lower[i-1] if i > 0 and not np.isnan(donchian_lower[i-1]) else np.nan
        
        # Bullish breakout: close above prior bar's upper channel
        bullish_breakout = (not np.isnan(prev_upper) and close[i] > prev_upper)
        
        # Bearish breakout: close below prior bar's lower channel
        bearish_breakout = (not np.isnan(prev_lower) and close[i] < prev_lower)
        
        # === MINIMUM HOLD: 2 bars ===
        min_hold = (i - entry_bar) >= 2
        
        # === EXITS ===
        if in_position:
            # Stop-loss: 2*ATR from entry
            if position_side > 0:
                stop_hit = low[i] < (entry_price - 2.0 * entry_atr)
            else:
                stop_hit = high[i] > (entry_price + 2.0 * entry_atr)
            
            # Exit on opposite breakout
            reversal_exit = (position_side > 0 and bearish_breakout) or \
                           (position_side < 0 and bullish_breakout)
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            elif min_hold and reversal_exit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            else:
                signals[i] = position_side * SIZE
        
        # === NEW POSITIONS ===
        if not in_position:
            # Skip if not trending
            if not is_trending:
                signals[i] = 0.0
                continue
            
            # LONG: Bullish breakout + volume spike + weekly bullish
            if bullish_breakout and vol_spike and weekly_bullish:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                signals[i] = SIZE
            
            # SHORT: Bearish breakout + volume spike + weekly bearish
            elif bearish_breakout and vol_spike and weekly_bearish:
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
    
    return signals