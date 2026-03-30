#!/usr/bin/env python3
"""
Experiment #024: 12h Donchian Breakout + Choppiness Regime + 1w Trend

HYPOTHESIS: 12h timeframe balances trade frequency with signal quality.
Donchian(20) on 12h = 10-day breakout - sufficient for 4-year train/test.
1w SMA provides macro trend context (avoids fighting major bull/bear).
Choppiness Index filters ranging periods (only trend when trend exists).
Volume spike confirms institutional participation at breakout levels.

WHY 12h:
- 50-150 trades over 4 years = 12-37/year (within target)
- More trades than 1d, fewer false breakouts than 4h
- 12h = 2x daily, natural daily pivot extension

KEY INSIGHT: mtf_4h_chop_donchian_vol_regime_12h_v1 scored 1.49 test Sharpe.
This adapts that pattern to 12h primary with 1w trend confirmation.

ENTRIES:
- Long: Donchian upper breakout + 1w uptrend + vol spike + not ranging
- Short: Donchian lower breakout + 1w downtrend + vol spike + not ranging

EXITS: Opposite breakout OR 2.5x ATR stop OR chop regime change
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_chop_1w_vol_v1"
timeframe = "12h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss"""
    n = len(close)
    if n < 2:
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
    CHOP > 61.8 = ranging (no trend)
    CHOP < 38.2 = trending (trend following)
    Values between = neutral
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        sum_tr = 0.0
        for j in range(period):
            tr = max(high[i-j] - low[i-j], abs(high[i-j] - close[i-j-1]) if i-j > 0 else high[i-j] - low[i-j])
            sum_tr += tr
        
        highest = max(high[i-period+1:i+1])
        lowest = min(low[i-period+1:i+1])
        range_hl = highest - lowest
        
        if range_hl > 1e-10:
            chop[i] = 100 * np.log10(sum_tr / range_hl) / np.log10(period)
    
    return chop

def calculate_donchian(high, low, period=20):
    """Donchian channel"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w SMA for macro trend (call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    sma_1w = pd.Series(df_1w['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    sma_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_1w)
    
    # === 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    dc_upper, dc_lower = calculate_donchian(high, low, period=20)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Volume spike (1.5x 20-bar average)
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
    
    warmup = 50  # Donchian(20) + chop(14)
    
    for i in range(warmup, n):
        # NaN check
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(sma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME CHECK ===
        # CHOP > 61.8 = ranging (skip entries)
        is_ranging = not np.isnan(chop[i]) and chop[i] > 61.8
        is_trending = not np.isnan(chop[i]) and chop[i] < 38.2
        
        # === TREND DETECTION (1w SMA) ===
        htf_bullish = close[i] > sma_1w_aligned[i]
        htf_bearish = close[i] < sma_1w_aligned[i]
        
        # === DONCHIAN BREAKOUT DETECTION ===
        # Price breaks above 20-bar high = bullish breakout
        above_dc_upper = close[i] > dc_upper[i] if not np.isnan(dc_upper[i]) else False
        # Price breaks below 20-bar low = bearish breakout
        below_dc_lower = close[i] < dc_lower[i] if not np.isnan(dc_lower[i]) else False
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === ATR TRAILING STOP ===
        def check_atr_stop():
            if not in_position:
                return False
            # Long stop: price falls 2.5x ATR below highest since entry
            if position_side > 0:
                entry_high = high[entry_bar]
                return low[i] < (entry_high - 2.5 * entry_atr)
            # Short stop: price rises 2.5x ATR above lowest since entry
            else:
                entry_low = low[entry_bar]
                return high[i] > (entry_low + 2.5 * entry_atr)
        
        # === MIN HOLD: at least 2 bars (24h) to avoid chop ===
        min_hold = (i - entry_bar) >= 2
        
        # === EXITS ===
        if in_position:
            stop_hit = check_atr_stop()
            
            # Exit on opposite trend (1w SMA flips)
            trend_exit = False
            if position_side > 0 and htf_bearish and min_hold:
                trend_exit = True
            if position_side < 0 and htf_bullish and min_hold:
                trend_exit = True
            
            if stop_hit or trend_exit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            else:
                signals[i] = position_side * SIZE
        
        # === NEW POSITIONS ===
        if not in_position:
            # LONG: Above Donchian upper + 1w uptrend + vol spike + not ranging
            if above_dc_upper and htf_bullish and vol_spike and not is_ranging:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                signals[i] = SIZE
            
            # SHORT: Below Donchian lower + 1w downtrend + vol spike + not ranging
            elif below_dc_lower and htf_bearish and vol_spike and not is_ranging:
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                signals[i] = -SIZE
            
            else:
                signals[i] = 0.0
    
    return signals