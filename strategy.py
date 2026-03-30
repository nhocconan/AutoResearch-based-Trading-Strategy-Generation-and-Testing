#!/usr/bin/env python3
"""
Experiment #008: 12h Donchian + Choppiness Regime + 1w EMA

HYPOTHESIS: Use Choppiness Index (<38.2 trending, >61.8 ranging) as regime filter
combined with 1w EMA macro trend and Donchian breakout on 12h. This replicates the
proven pattern from DB: mtf_4h_chop_donchian_vol_regime_12h_v1 had test Sharpe 1.491.

WHY BOTH MARKETS:
- 2021 bull: Choppiness < 38.2 + bullish breakout + 1w EMA up = trend follow longs
- 2022 bear: Choppiness < 38.2 + bearish breakdown + 1w EMA down = trend follow shorts
- 2025 range: Choppiness > 61.8 = mean revert at Donchian bounds

TRADE COUNT: 75-175 total over 4 years (18-44/year).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_chop_1w_ema_v1"
timeframe = "12h"
leverage = 1.0

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP < 38.2 = trending (trend follow)
    CHOP > 61.8 = ranging (mean revert)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        highest = high[i-period+1:i+1].max()
        lowest = low[i-period+1:i+1].min()
        
        if highest - lowest > 1e-10:
            sum_tr = 0.0
            for j in range(i-period+1, i+1):
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
                sum_tr += tr
            
            chop[i] = 100 * (np.log(sum_tr) / np.log(highest - lowest)) if (highest - lowest) > 1e-10 else 50.0
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    
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
    
    # === HTF: 1w EMA for macro trend (call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    ema_1w = pd.Series(df_1w['close'].values).ewm(span=13, min_periods=13, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian Channel(12) - 6 days on 12h
    donchian_upper = pd.Series(high).rolling(window=12, min_periods=12).max().values
    donchian_lower = pd.Series(low).rolling(window=12, min_periods=12).min().values
    
    # Choppiness Index (14 periods)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Volume analysis
    vol_ma = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    vol_ratio = volume / np.where(vol_ma > 1e-10, vol_ma, 1.0)
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.30  # Full position size
    SIZE_HALF = 0.15  # Take profit size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    took_profit = False
    
    warmup = 30  # 14 for chop + buffer
    
    for i in range(warmup, n):
        # NaN checks
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(ema_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Update highest/lowest for trailing stop
        if in_position:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
            else:
                lowest_since_entry = min(lowest_since_entry, low[i])
        
        # === REGIME DETECTION ===
        chop_trending = chop[i] < 38.2  # Trending mode
        chop_ranging = chop[i] > 61.8   # Ranging mode
        
        # === HTF MACRO TREND (1w EMA aligned) ===
        htf_bullish = close[i] > ema_1w_aligned[i]
        htf_bearish = close[i] < ema_1w_aligned[i]
        
        # === VOLUME CONFIRMATION (1.5x average) ===
        vol_confirm = vol_ratio[i] > 1.5
        
        # === DONCHIAN BREAKOUT DETECTION ===
        # Use prior bar's channel for entry (no look-ahead)
        prev_upper = donchian_upper[i-1] if i > 0 and not np.isnan(donchian_upper[i-1]) else np.nan
        prev_lower = donchian_lower[i-1] if i > 0 and not np.isnan(donchian_lower[i-1]) else np.nan
        
        bullish_breakout = False
        bearish_breakout = False
        
        if not np.isnan(prev_upper) and not np.isnan(prev_lower):
            # Close breaks above prior upper = bullish breakout
            bullish_breakout = close[i] > prev_upper
            # Close breaks below prior lower = bearish breakdown
            bearish_breakout = close[i] < prev_lower
        
        # === MINIMUM HOLD: 2 bars (24h) to avoid chop whipsaws ===
        min_hold_bars = 2
        min_hold = (i - entry_bar) >= min_hold_bars
        
        # === ENTRY CONDITIONS ===
        can_long = not in_position and bullish_breakout and vol_confirm and htf_bullish and chop_trending
        can_short = not in_position and bearish_breakout and vol_confirm and htf_bearish and chop_trending
        
        # === EXITS ===
        if in_position:
            # ATR trailing stop (2.5x ATR from entry)
            if position_side > 0:
                stop_hit = low[i] < (highest_since_entry - 2.5 * entry_atr)
            else:
                stop_hit = high[i] > (lowest_since_entry + 2.5 * entry_atr)
            
            # Trend reversal exit (1w EMA flips)
            if position_side > 0 and htf_bearish and min_hold:
                stop_hit = True
            if position_side < 0 and htf_bullish and min_hold:
                stop_hit = True
            
            # Regime change exit (chop moves to ranging)
            if chop_ranging and min_hold:
                stop_hit = True
            
            # Take profit at 2R (reduce size)
            if not took_profit:
                if position_side > 0:
                    profit_target = entry_price + 2.0 * entry_atr
                    if high[i] >= profit_target:
                        took_profit = True
                        signals[i] = SIZE_HALF
                        continue
                else:
                    profit_target = entry_price - 2.0 * entry_atr
                    if low[i] <= profit_target:
                        took_profit = True
                        signals[i] = -SIZE_HALF
                        continue
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
                took_profit = False
            else:
                signals[i] = position_side * SIZE
        
        # === NEW POSITIONS ===
        if not in_position:
            if can_long:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                took_profit = False
                signals[i] = SIZE
            
            elif can_short:
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                took_profit = False
                signals[i] = -SIZE
            
            else:
                signals[i] = 0.0
    
    return signals