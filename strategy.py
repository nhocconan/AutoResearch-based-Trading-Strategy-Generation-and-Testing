#!/usr/bin/env python3
"""
Experiment #025: 1d Donchian Breakout + Weekly Trend + Volume

HYPOTHESIS: Simple Donchian(20) breakout on 1d with weekly EMA50 trend filter
and volume confirmation. This matches proven DB pattern (mtf_4h_donchian_volume_ema
simplified: Sharpe=0.092). Weekly trend filter aligns entries with higher timeframe
direction, reducing whipsaws. Volume spike confirms institutional participation.
1d timeframe naturally limits trades to 50-100 total over 4 years (12-25/year).

WHY IT WORKS IN BULL + BEAR:
- Bull: Breakout above 20d high + price>1w EMA50 + volume = strong continuation
- Bear: Breakdown below 20d low + price<1w EMA50 + volume = short momentum
- Range: Choppiness>61.8 filters out sideways markets

TARGET: 50-100 total trades over 4 years.
Signal size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_weekly_trend_v1"
timeframe = "1d"
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - values > 61.8 = choppy/range, < 38.2 = trending"""
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j])
            atr_sum += tr
        
        hh = max(high[i - period + 1:i + 1])
        ll = min(low[i - period + 1:i + 1])
        range_sum = hh - ll
        
        if range_sum > 0:
            chop[i] = 100 * (np.log10(atr_sum / range_sum) / np.log10(period))
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA50 for trend direction
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === Local 1d indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    donchian_up, donchian_lo = calculate_donchian(high, low, period=20)
    
    # Volume ratio (20-period MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    
    warmup = 50  # Donchian(20) needs 20, chop needs 14, weekly EMA50 needs 50
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(ema_50_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === HTF TREND: Weekly EMA50 ===
        above_weekly_ema = close[i] > ema_50_aligned[i]
        below_weekly_ema = close[i] < ema_50_aligned[i]
        
        # === CHOPPINESS REGIME FILTER ===
        chop = chop_14[i]
        is_choppy = chop > 61.8 if not np.isnan(chop) else False
        is_trending = chop < 50 if not np.isnan(chop) else True
        
        # Skip in choppy markets
        if is_choppy:
            if in_position:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            else:
                signals[i] = 0.0
            continue
        
        # === DONCHIAN BREAKOUT (use shift(1) to avoid look-ahead) ===
        if i >= 21:
            donchian_broken_up = close[i] > donchian_up[i - 1]
            donchian_broken_down = close[i] < donchian_lo[i - 1]
        else:
            donchian_broken_up = False
            donchian_broken_down = False
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Weekly trend up + Donchian breakout + volume ===
            if above_weekly_ema and donchian_broken_up and vol_spike:
                desired_signal = SIZE
            
            # === SHORT: Weekly trend down + Donchian breakdown + volume ===
            elif below_weekly_ema and donchian_broken_down and vol_spike:
                desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR trailing) ===
        if in_position:
            if position_side > 0:
                # Long: trail stop at lowest low - 0.5 ATR buffer
                lowest_since_entry = np.min(low[entry_bar:i+1])
                stop_price = lowest_since_entry - 0.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Short: trail stop at highest high + 0.5 ATR buffer
                highest_since_entry = np.max(high[entry_bar:i+1])
                stop_price = highest_since_entry + 0.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 2 bars to avoid fee churn ===
        if in_position and (i - entry_bar) < 2:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals