#!/usr/bin/env python3
"""
Experiment #023: 12h Donchian(12) + 1d SMA50 Trend + Volume Spike

HYPOTHESIS: Multi-timeframe trend following with structural breakout.
- 1d SMA50: proven trend filter (filters bear traps in 2022)
- 12h Donchian(12): ~36 raw breakouts/yr after 12h filter
- Volume spike: institutional confirmation
- CHOP regime: avoid range-bound choppy periods

WHY IT SHOULD WORK IN BOTH MARKETS:
- Bull market: price > SMA50 + breakout = follow the trend
- Bear market (2022): price < SMA50 = no longs, short breakouts only
- Range (2025): CHOP filter avoids false breakouts

TRADE COUNT ESTIMATE:
- 12h bars/4yr ≈ 8760/2 = 2920 bars
- Donchian(12) breakout: ~1 per 40 bars = ~73 raw signals
- SMA50 trend filter: ~50% pass = ~36 signals
- Volume spike: ~60% pass = ~22 trades/symbol (LOW!)
- CHOP < 60 filter adds: ~70% pass = ~25-30 trades

NEED TO LOOSEN: Use Donchian(8) to get more breakouts
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_1d_sma50_vol_v1"
timeframe = "12h"
leverage = 1.0

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

def calculate_chop(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - vectorized
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        period_high = high[i-period+1:i+1].max()
        period_low = low[i-period+1:i+1].min()
        
        if period_high > period_low:
            sum_tr = 0.0
            for j in range(i-period+1, i+1):
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j])
                sum_tr += tr
            
            if period_high != period_low:
                chop[i] = 100 * np.log10(sum_tr / (period_high - period_low)) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d SMA50 for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values.astype(np.float64)
    sma50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    # Align to 12h (shift by 1 to avoid look-ahead)
    sma50_aligned = align_htf_to_ltf(prices, df_1d, sma50_1d)
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_chop(high, low, close, period=14)
    
    # Donchian Channel(8) - price structure for 12h
    donchian_upper = pd.Series(high).rolling(window=8, min_periods=8).max().values
    donchian_lower = pd.Series(low).rolling(window=8, min_periods=8).min().values
    
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
    
    warmup = 60  # Enough for Donchian(8), ATR14, CHOP14, 1d SMA50
    
    for i in range(warmup, n):
        # NaN checks
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(sma50_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME FILTER: CHOP < 58 (not extremely choppy) ===
        choppy_market = chop[i] > 58.0
        
        # === TREND FILTER: 1d SMA50 ===
        # Bull trend: price > SMA50
        bull_trend = close[i] > sma50_aligned[i]
        # Bear trend: price < SMA50
        bear_trend = close[i] < sma50_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.4
        
        # === DONCHIAN(8) BREAKOUT (prior bar's range) ===
        prev_upper = donchian_upper[i-1] if i > 0 and not np.isnan(donchian_upper[i-1]) else np.nan
        prev_lower = donchian_lower[i-1] if i > 0 and not np.isnan(donchian_lower[i-1]) else np.nan
        
        # Bullish breakout: close above prior bar's upper channel
        bullish_breakout = (not np.isnan(prev_upper) and close[i] > prev_upper)
        
        # Bearish breakout: close below prior bar's lower channel
        bearish_breakout = (not np.isnan(prev_lower) and close[i] < prev_lower)
        
        # === MINIMUM HOLD: 3 bars (~1.5 days) ===
        min_hold = (i - entry_bar) >= 3
        
        # === EXITS ===
        if in_position:
            # Stop-loss: 2.5 ATR from entry
            if position_side > 0:
                stop_hit = low[i] < (entry_price - 2.5 * entry_atr)
            else:
                stop_hit = high[i] > (entry_price + 2.5 * entry_atr)
            
            # Exit on opposite breakout (trend reversal)
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
            # Skip if choppy market
            if choppy_market:
                signals[i] = 0.0
                continue
            
            # LONG: Bull trend + Bullish breakout + volume spike
            if bullish_breakout and vol_spike and bull_trend:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                signals[i] = SIZE
            
            # SHORT: Bear trend + Bearish breakout + volume spike
            elif bearish_breakout and vol_spike and bear_trend:
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
    
    return signals