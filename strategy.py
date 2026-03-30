#!/usr/bin/env python3
"""
Experiment #022: 1d TRIX Momentum + Weekly Trend + Volume Spike

HYPOTHESIS: TRIX is a momentum oscillator that generates fewer but higher-quality
signals than EMA crosses. Combined with weekly trend filter, this creates:
1. Weekly EMA(21) as trend direction filter - only trade with weekly trend
2. Daily TRIX(14) crossover for momentum entry
3. Volume spike confirmation (2.0x 20d MA) - avoid fake breakouts
4. Choppiness < 55 as regime filter - only enter trending markets

WHY IT WORKS IN BULL + BEAR + RANGE:
- Bull: Weekly up + TRIX bullish cross + volume = strong entry
- Bear: Weekly down + TRIX bearish cross + volume = strong short
- Range: CHOP > 55 = skip entirely (avoids whipsaws)
- ATR stoploss scales to volatility, handles 2022 crash

TARGET: 60-120 total trades over 4 years (15-30/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_trix_weekly_vol_chop_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_trix(close, period=14):
    """TRIX: Triple smoothed EMA rate of change"""
    n = len(close)
    if n < period * 3:
        return np.full(n, np.nan)
    
    # Triple EMA
    ema1 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    ema2 = pd.Series(ema1).ewm(span=period, min_periods=period, adjust=False).mean().values
    ema3 = pd.Series(ema2).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Rate of change of triple EMA
    trix = np.full(n, np.nan)
    for i in range(period * 3, n):
        if ema3[i - period] != 0 and not np.isnan(ema3[i - period]):
            trix[i] = 100 * (ema3[i] - ema3[i - period]) / ema3[i - period]
    
    return trix

def calculate_trix_signal(trix, period=9):
    """TRIX signal line (EMA of TRIX)"""
    n = len(trix)
    signal = pd.Series(trix).ewm(span=period, min_periods=period, adjust=False).mean().values
    return signal

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

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index: <45 trending, >61.8 ranging"""
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
    
    # === Load HTF data ONCE before loop ===
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA(21) for trend direction
    ema_21_1w = pd.Series(df_1w['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # === Daily indicators ===
    trix = calculate_trix(close, period=14)
    trix_signal = calculate_trix_signal(trix, period=9)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Volume ratio (20-period MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Donchian channel (20-period) for structure
    channel_up = pd.Series(high).rolling(window=20, min_periods=20).max().values
    channel_lo = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    entry_trix = 0.0
    
    warmup = 100  # TRIX needs ~42 bars, add buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(trix[i]) or np.isnan(trix_signal[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(ema_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(channel_up[i]) or np.isnan(channel_lo[i]):
            signals[i] = 0.0
            continue
        
        # === WEEKLY TREND FILTER ===
        weekly_trend_up = close[i] > ema_aligned[i]
        weekly_trend_down = close[i] < ema_aligned[i]
        
        # === CHOPPINESS REGIME FILTER ===
        is_trending = chop[i] < 55
        is_choppy = chop[i] > 61.8
        
        # === TRIX CROSSOVER SIGNALS ===
        # Bullish: TRIX crosses above signal
        trix_bullish_cross = trix[i] > trix_signal[i] and trix[i-1] <= trix_signal[i-1]
        # Bearish: TRIX crosses below signal
        trix_bearish_cross = trix[i] < trix_signal[i] and trix[i-1] >= trix_signal[i-1]
        
        # Strong momentum: TRIX far from signal line (confirming trend strength)
        trix_bullish_strong = trix[i] > 0.5 and trix_signal[i] > 0
        trix_bearish_strong = trix[i] < -0.5 and trix_signal[i] < 0
        
        # === VOLUME CONFIRMATION (2.0x) ===
        vol_spike = vol_ratio[i] > 2.0
        
        # === PRICE STRUCTURE (near channel boundary) ===
        range_size = channel_up[i] - channel_lo[i]
        price_near_high = close[i] > channel_up[i] - range_size * 0.2  # within 20% of high
        price_near_low = close[i] < channel_lo[i] + range_size * 0.2    # within 20% of low
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Weekly up + TRIX bullish cross + volume + trending + price near high ===
            long_conditions = (
                weekly_trend_up and
                (trix_bullish_cross or trix_bullish_strong) and
                vol_spike and
                is_trending and
                price_near_high
            )
            if long_conditions:
                desired_signal = SIZE
            
            # === SHORT: Weekly down + TRIX bearish cross + volume + trending + price near low ===
            short_conditions = (
                weekly_trend_down and
                (trix_bearish_cross or trix_bearish_strong) and
                vol_spike and
                is_trending and
                price_near_low
            )
            if short_conditions:
                desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR) ===
        if in_position:
            if position_side > 0:
                # Long: stop if price falls 2.5 ATR from entry
                stop_price = entry_price - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Exit if weekly trend flips
                if weekly_trend_down:
                    desired_signal = 0.0
                
                # Exit if market becomes choppy
                if is_choppy:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Short: stop if price rises 2.5 ATR from entry
                stop_price = entry_price + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Exit if weekly trend flips
                if weekly_trend_up:
                    desired_signal = 0.0
                
                # Exit if market becomes choppy
                if is_choppy:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 3 bars to avoid fee churn ===
        if in_position and (i - entry_bar) < 3:
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
                entry_trix = trix[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals