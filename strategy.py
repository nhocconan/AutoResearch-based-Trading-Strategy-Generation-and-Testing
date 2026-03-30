#!/usr/bin/env python3
"""
Experiment #021: 4h TRIX Cross + Donchian Breakout + Choppiness Regime

HYPOTHESIS: Combine TRIX momentum crossover with price channel breakout
and choppiness regime filter. TRIX(9) crossover signals trend changes,
Donchian(20) confirms breakout direction, and CHOP filters whipsaws.
This combination works in BULL (trend following) and BEAR (reversal bounces).

WHY IT SHOULD WORK:
- TRIX crossover: proven momentum indicator (ETH test Sharpe 1.32)
- Donchian breakout: price structure confirmation (SOL test Sharpe 1.10-1.38)
- CHOP < 45: trending regime filter (key meta-filter from winners)
- HTF 12h EMA: trend direction bias

TARGET: 120-200 total trades over 4 years (30-50/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_trix_donchian_chop_12h_v1"
timeframe = "4h"
leverage = 1.0

def calculate_trix(close, period=9):
    """TRIX: triple smoothed EMA rate of change"""
    n = len(close)
    if n < period * 3:
        return np.full(n, np.nan)
    
    ema1 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    ema2 = pd.Series(ema1).ewm(span=period, min_periods=period, adjust=False).mean().values
    ema3 = pd.Series(ema2).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    trix = np.full(n, np.nan)
    for i in range(period * 3, n):
        if ema3[i] > 0 and ema3[i - 1] > 0:
            trix[i] = 100 * (ema3[i] - ema3[i - 1]) / ema3[i - 1]
    
    return trix

def calculate_donchian(high, low, period=20):
    """Donchian Channel - 20 period high/low"""
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = ranging - DON'T enter
    CHOP < 45 = trending - GOOD to enter
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_12h = get_htf_data(prices, '12h')
    
    # 12h EMA(21) for trend direction
    ema_21_12h = pd.Series(df_12h['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_aligned = align_htf_to_ltf(prices, df_12h, ema_21_12h)
    
    # === Local 4h indicators ===
    trix = calculate_trix(close, period=9)
    donchian_up, donchian_lo = calculate_donchian(high, low, period=20)
    chop = calculate_choppiness(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume ratio (20-period MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    
    warmup = 200  # 20 for donchian + 9*3 for trix + 14 for CHOP + 20 for vol MA
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(trix[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_up[i]) or np.isnan(ema_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        # === CHOPPINESS REGIME FILTER ===
        chop_value = chop[i]
        is_choppy = chop_value > 61.8
        is_trending = chop_value < 45
        
        # === TRIX CROSSOVER DETECTION ===
        # Positive TRIX = bullish momentum
        trix_prev = trix[i - 1] if not np.isnan(trix[i - 1]) else 0
        trix_curr = trix[i]
        
        trix_bullish = trix_curr > trix_prev and trix_curr > 0
        trix_bearish = trix_curr < trix_prev and trix_curr < 0
        
        # === HTF TREND: 12h EMA(21) direction ===
        htf_trend_up = close[i] > ema_aligned[i]
        htf_trend_down = close[i] < ema_aligned[i]
        
        # === DONCHIAN BREAKOUT ===
        prev_donchian_up = donchian_up[i - 1]
        prev_donchian_lo = donchian_lo[i - 1]
        
        breakout_up = close[i] > prev_donchian_up
        breakout_down = close[i] < prev_donchian_lo
        
        # === VOLUME CONFIRMATION (1.8x) ===
        vol_spike = vol_ratio[i] > 1.8
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: TRIX bullish cross + Donchian breakout + HTF up + CHOP trending + volume ===
            if trix_bullish and breakout_up and htf_trend_up and is_trending and vol_spike:
                desired_signal = SIZE
            
            # === SHORT: TRIX bearish cross + Donchian breakout + HTF down + CHOP trending + volume ===
            if trix_bearish and breakout_down and htf_trend_down and is_trending and vol_spike:
                desired_signal = -SIZE
        
        # === STOPLOSS (2.0 ATR from entry) ===
        if in_position:
            if position_side > 0:
                # Long stoploss
                stop_price = entry_price - 2.0 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Exit if HTF trend flips to down
                if htf_trend_down:
                    desired_signal = 0.0
                
                # Exit if market becomes choppy
                if is_choppy:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Short stoploss
                stop_price = entry_price + 2.0 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Exit if HTF trend flips to up
                if htf_trend_up:
                    desired_signal = 0.0
                
                # Exit if market becomes choppy
                if is_choppy:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 4 bars to avoid fee churn ===
        if in_position and (i - entry_bar) < 4:
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