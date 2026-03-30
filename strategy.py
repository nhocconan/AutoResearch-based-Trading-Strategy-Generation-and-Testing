#!/usr/bin/env python3
"""
Experiment #024: 4h Camarilla Pivot + Choppiness Regime + Volume Spike

HYPOTHESIS: Camarilla S3/R3 levels are statistically significant reversal
points. When price reaches these extremes in a "choppy" (non-trending) regime,
mean reversion is more likely. In trending regimes, we skip entries to avoid
chasing. Volume spike confirms the reversal is starting.

WHY IT WORKS IN BULL AND BEAR:
- 2021 bull: Buy S3 bounces in uptrends (1d SMA confirms uptrend)
- 2022 bear: Sell R3 bounces in downtrends (1d SMA confirms downtrend)
- 2023-2024 range: Choppiness filter keeps us flat in ranging markets
- 2025 bear: Short bounces at R3 when 1d downtrend confirmed

KEY INSIGHT: Top DB performer "gen_camarilla_pivot_volume_spike_choppiness_4h_v1"
had test_sharpe=1.471 with 95 trades. This strategy is a tighter implementation
of that proven pattern.

REFINEMENT from #007: Lower timeframe (4h), add choppiness filter, remove
S4/R4 extremes (they're too aggressive), tighten entry window.
Target: 80-150 total trades over 4 years (20-37/year).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_camarilla_chop_vol_v1"
timeframe = "4h"
leverage = 1.0

def calculate_camarilla(high, low, close):
    """
    Camarilla pivot levels (classic):
    R3 = close + (high - low) * 1.1
    R4 = close + (high - low) * 1.1 / 2 = close + rng * 1.1/2
    S3 = close - (high - low) * 1.1
    S4 = close - (high - low) * 1.1 / 2
    """
    rng = high - low
    r3 = close + rng * 1.1
    r4 = close + rng * 1.1 / 2.0  # Classic uses 1.1/2 for R4/S4
    s3 = close - rng * 1.1
    s4 = close - rng * 1.1 / 2.0
    return r3, r4, s3, s4

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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP): measures market choppiness/trending
    CHOP > 61.8 = choppy/mean reverting
    CHOP < 38.2 = trending
    Range: 0-100, lower = more trending
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        # Sum of true range over period
        tr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr_sum += max(high[j] - low[j], abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j])
        
        # Highest - lowest over period
        hh = max(high[i - period + 1:i + 1])
        ll = min(low[i - period + 1:i + 1])
        
        if hh - ll > 0:
            chop[i] = 100 * np.log10(tr_sum / (hh - ll)) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d SMA for macro trend (call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    sma_1d_50 = pd.Series(df_1d['close'].values).rolling(window=50, min_periods=50).mean().values
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d_50)
    
    # === 4h indicators (computed once) ===
    atr_14 = calculate_atr(high, low, close, period=14)
    r3, r4, s3, s4 = calculate_camarilla(high, low, close)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Volume analysis
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
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 60
    
    for i in range(warmup, n):
        # NaN check
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(sma_1d_aligned[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # Update highest/lowest for trailing stop
        if in_position:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
            else:
                lowest_since_entry = min(lowest_since_entry, low[i])
        
        # === TREND DETECTION ===
        # 1d SMA for macro direction
        htf_bullish = close[i] > sma_1d_aligned[i]
        htf_bearish = close[i] < sma_1d_aligned[i]
        
        # === CHOPPINESS REGIME ===
        # CHOP > 61.8 = choppy (good for mean reversion/Camarilla)
        # CHOP < 38.2 = trending (skip - trend following is better)
        chop_choppy = chop[i] > 61.8
        chop_trending = chop[i] < 38.2
        
        # === CAMARILLA SIGNALS (tight entry) ===
        # Price at S3 level with 1.5% tolerance
        at_s3 = (close[i] <= s3[i] * 1.015) and (close[i] >= s3[i] * 0.985)
        # Price at R3 level with 1.5% tolerance
        at_r3 = (close[i] >= r3[i] * 0.985) and (close[i] <= r3[i] * 1.015)
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === MINIMUM HOLD: 2 bars (8h) ===
        min_hold = (i - entry_bar) >= 2
        
        # === ATR TRAILING STOP (2.5x ATR) ===
        if in_position:
            if position_side > 0:
                stop_hit = low[i] < (highest_since_entry - 2.5 * entry_atr)
                # Trend reversal exit
                if htf_bearish and min_hold:
                    stop_hit = True
            else:
                stop_hit = high[i] > (lowest_since_entry + 2.5 * entry_atr)
                # Trend reversal exit
                if htf_bullish and min_hold:
                    stop_hit = True
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            else:
                signals[i] = position_side * SIZE
            continue
        
        # === NEW POSITIONS ===
        # LONG: Price at S3 + volume spike + 1d uptrend + choppy regime
        if at_s3 and vol_spike and htf_bullish and chop_choppy:
            in_position = True
            position_side = 1
            entry_price = close[i]
            entry_atr = atr_14[i]
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        
        # SHORT: Price at R3 + volume spike + 1d downtrend + choppy regime
        elif at_r3 and vol_spike and htf_bearish and chop_choppy:
            in_position = True
            position_side = -1
            entry_price = close[i]
            entry_atr = atr_14[i]
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        
        else:
            signals[i] = 0.0
    
    return signals