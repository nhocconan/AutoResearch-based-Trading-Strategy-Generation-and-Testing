#!/usr/bin/env python3
"""
Experiment #021: 12h Donchian + 1d HMA Trend + Volume + Min Hold

HYPOTHESIS:
- 12h timeframe naturally limits trades to ~20-50/year (80-200 over 4 years)
- 1d HMA for HTF trend direction (smoother than EMA, proven in DB winners)
- Donchian(20) on 12h for structure (same as top performer mtf_4h_chop_donchian_vol_regime_12h_v1)
- Volume spike confirmation (1.8x) to filter false breakouts
- CHOP < 50 to skip ranging markets
- MINIMUM HOLD of 10 bars to prevent signal churn
- ATR(14) stoploss scaled to volatility

WHY IT SHOULD WORK IN BULL + BEAR + RANGE:
- Bull: Price breaks Donchian high + 1d HMA rising + vol spike → long
- Bear: Price breaks Donchian low + 1d HMA falling + vol spike → short
- Range: CHOP > 50 → SKIP entirely (avoids whipsaws, the #1 killer)
- ATR-based stoploss scales with volatility (handles 2022 crash naturally)

KEY IMPROVEMENT over failed #011 (54 trades, Sharpe -0.800):
- Use 1d HMA instead of weekly EMA (smoother, more responsive)
- Loosen entry slightly (1.8x vol instead of 2.0x)
- Enforce minimum 10-bar hold to reduce churn
- CHOP < 50 instead of < 45
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_hma_vol_chop_1d_v1"
timeframe = "12h"
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

def calculate_hma(data, period):
    """Hull Moving Average"""
    n = len(data)
    if n < period:
        return np.full(n, np.nan)
    
    half_len = period // 2
    half = pd.Series(data).rolling(window=half_len, min_periods=half_len).mean()
    full = pd.Series(data).rolling(window=period, min_periods=period).mean()
    hma = 2 * half - full
    return hma.values

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = ranging - DON'T enter
    CHOP < 50 = trending - GOOD to enter
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan)
    for i in range(period, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        
        if highest > lowest and atr_sum > 0:
            range_hl = highest - lowest
            chop[i] = 100 * np.log10(atr_sum / range_hl) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load 1d HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d HMA(21) for trend direction
    hma_21_1d = calculate_hma(df_1d['close'].values, 21)
    hma_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    
    # 1d HMA(48) for longer-term trend
    hma_48_1d = calculate_hma(df_1d['close'].values, 48)
    hma48_aligned = align_htf_to_ltf(prices, df_1d, hma_48_1d)
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Donchian channels (20-period = 10 days on 12h)
    donchian_up = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lo = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume ratio (20-period MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size - moderate for 12h
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    
    warmup = 250  # 200 for Donchian + 14 for CHOP + 20 for vol MA + margin
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_aligned[i]) or np.isnan(hma48_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_up[i]) or np.isnan(donchian_lo[i]):
            signals[i] = 0.0
            continue
        
        # === CHOPPINESS REGIME FILTER ===
        chop_value = chop[i]
        is_trending = chop_value < 50  # trending market
        is_ranging = chop_value > 61.8  # skip ranging
        
        # === 1d HTF TREND (HMA direction) ===
        hma_trend_up = hma_aligned[i] > hma_aligned[i - 1] if i > 0 else True
        hma48_trend_up = hma48_aligned[i] > hma48_aligned[i - 1] if i > 0 else True
        htf_bull = hma_trend_up and hma48_trend_up  # both short and long HMA rising
        
        hma_trend_down = hma_aligned[i] < hma_aligned[i - 1] if i > 0 else False
        hma48_trend_down = hma48_aligned[i] < hma48_aligned[i - 1] if i > 0 else False
        htf_bear = hma_trend_down and hma48_trend_down  # both HMA falling
        
        # === VOLUME CONFIRMATION (1.8x, moderate) ===
        vol_spike = vol_ratio[i] > 1.8
        
        # === DONCHIAN BREAKOUT ===
        prev_donchian_up = donchian_up[i - 1]
        prev_donchian_lo = donchian_lo[i - 1]
        
        breakout_up = close[i] > prev_donchian_up
        breakout_down = close[i] < prev_donchian_lo
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # Only enter in trending markets
            if is_ranging:
                signals[i] = 0.0
                continue
            
            # === LONG: Breakout up + HTF bull + volume spike ===
            if breakout_up and htf_bull and vol_spike and is_trending:
                desired_signal = SIZE
            
            # === SHORT: Breakout down + HTF bear + volume spike ===
            if breakout_down and htf_bear and vol_spike and is_trending:
                desired_signal = -SIZE
        
        # === STOPLOSS (3 ATR from entry - conservative for 12h) ===
        if in_position:
            if position_side > 0:
                # Long stoploss
                stop_price = entry_price - 3.0 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Exit if market becomes ranging
                if is_ranging:
                    desired_signal = 0.0
                
                # Exit if HTF trend flips to bear
                if htf_bear:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Short stoploss
                stop_price = entry_price + 3.0 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Exit if market becomes ranging
                if is_ranging:
                    desired_signal = 0.0
                
                # Exit if HTF trend flips to bull
                if htf_bull:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 10 bars to avoid fee churn ===
        bars_held = i - entry_bar
        if in_position and bars_held < 10:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or direction flip
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