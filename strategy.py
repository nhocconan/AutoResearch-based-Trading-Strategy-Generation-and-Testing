#!/usr/bin/env python3
"""
Experiment #009: 4h Donchian Breakout + Volume + 1d HMA Trend

HYPOTHESIS: Simple breakout strategy that works in both bull and bear.
- Bull markets: price breaks above Donchian resistance → continue higher
- Bear markets: rallies to upper band fail → short the failure
- Volume spike confirms institutional involvement
- 1d HMA filters entries to trend direction
- ATR stoploss protects against volatility

WHY THIS SHOULD WORK:
- Donchian breakouts are a proven price action pattern
- In bull: momentum continues after resistance break
- In bear: rallies fail at resistance = high-probability shorts
- Volume filter eliminates false breakouts
- ATR stop prevents blowup on volatility spikes

TARGET TRADES: 75-150 total over 4 years (proven range from DB)
DB REFERENCE: mtf_4h_hma_donchian_volume_rsi_12h_atr_v1 (Sharpe 1.38)

KEY DESIGN:
1. 4h Donchian(20) upper/lower as breakout zones
2. Volume > 1.5x 20-bar MA for confirmation
3. 1d HMA(21) rising/falling for trend filter
4. ATR(14) stoploss at 2.5x
5. Signal: 0.30 (discrete)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_vol_hma_simple_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - precompute once"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    # Weights for WMA
    weights = np.arange(1, half + 1, dtype=np.float64)
    half_weight_sum = np.sum(weights)
    
    full_weights = np.arange(1, period + 1, dtype=np.float64)
    full_weight_sum = np.sum(full_weights)
    
    sqrt_weights = np.arange(1, sqrt_n + 1, dtype=np.float64)
    sqrt_weight_sum = np.sum(sqrt_weights)
    
    result = np.full(n, np.nan, dtype=np.float64)
    wma_half = np.full(n, np.nan, dtype=np.float64)
    wma_full = np.full(n, np.nan, dtype=np.float64)
    
    # Half WMA
    for i in range(half - 1, n):
        window = close[i - half + 1:i + 1]
        if not np.any(np.isnan(window)):
            wma_half[i] = np.sum(window * weights) / half_weight_sum
    
    # Full WMA
    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
        if not np.any(np.isnan(window)):
            wma_full[i] = np.sum(window * weights) / full_weight_sum
    
    # Diff
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    # HMA of diff
    for i in range(sqrt_n - 1, n):
        window = diff[i - sqrt_n + 1:i + 1]
        if not np.any(np.isnan(window)):
            result[i] = np.sum(window * sqrt_weights) / sqrt_weight_sum
    
    return result

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
    
    # ============================================================
    # STEP 1: Load HTF data ONCE before loop
    # ============================================================
    df_1d = get_htf_data(prices, '1d')
    
    # 1d HMA(21) for trend - aligned to 4h
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # ============================================================
    # STEP 2: Pre-compute all 4h indicators (vectorized)
    # ============================================================
    
    # Donchian Channel (20 periods = 80 hours = 5 trading days)
    donchian_upper = pd.Series(close).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(close).rolling(window=20, min_periods=20).min().values
    
    # ATR for stoploss
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # ============================================================
    # STEP 3: Signal generation loop
    # ============================================================
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    position_side = 0      # 1=long, -1=short, 0=flat
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 50  # Need at least 20 bars for Donchian
    
    for i in range(warmup, n):
        # ============================================================
        # SANITY CHECKS: skip if indicators not ready
        # ============================================================
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(atr_14[i]) or atr_14[i] <= 0:
            continue
        if np.isnan(vol_ratio[i]):
            continue
        if np.isnan(hma_1d_aligned[i]):
            continue
        
        # ============================================================
        # STOPLOSS CHECK (check before new entries)
        # ============================================================
        if position_side != 0:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.5 * entry_atr
                if low[i] <= stop_price:
                    position_side = 0
                    entry_price = 0.0
                    entry_atr = 0.0
                    highest_since_entry = 0.0
                    lowest_since_entry = 0.0
                    continue
            
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.5 * entry_atr
                if high[i] >= stop_price:
                    position_side = 0
                    entry_price = 0.0
                    entry_atr = 0.0
                    highest_since_entry = 0.0
                    lowest_since_entry = 0.0
                    continue
        
        # ============================================================
        # ENTRY CONDITIONS
        # ============================================================
        desired_signal = 0.0
        
        # 1d HMA trend (need previous value for direction)
        prev_hma = hma_1d_aligned[i-1] if i > warmup else hma_1d_aligned[i]
        hma_rising = hma_1d_aligned[i] > prev_hma if not np.isnan(prev_hma) else True
        hma_falling = hma_1d_aligned[i] < prev_hma if not np.isnan(prev_hma) else False
        
        # Volume confirmation (>1.5x average)
        vol_confirm = vol_ratio[i] > 1.5
        
        # Price position relative to Donchian
        price_above_upper = close[i] > donchian_upper[i]
        price_below_lower = close[i] < donchian_lower[i]
        
        # ============================================================
        # LONG ENTRY: Breakout above + rising trend + volume
        # ============================================================
        if position_side == 0:
            if price_above_upper and hma_rising and vol_confirm:
                desired_signal = SIZE
            elif price_below_lower and hma_falling and vol_confirm:
                desired_signal = -SIZE
        
        # ============================================================
        # TAKE PROFIT at opposite band
        # ============================================================
        if position_side > 0:
            # TP when price hits lower band (mean reversion)
            if low[i] <= donchian_lower[i]:
                position_side = 0
                desired_signal = 0.0
        
        if position_side < 0:
            # TP when price hits upper band
            if high[i] >= donchian_upper[i]:
                position_side = 0
                desired_signal = 0.0
        
        # ============================================================
        # EXECUTE TRADE
        # ============================================================
        if desired_signal != 0.0:
            new_side = int(np.sign(desired_signal))
            if new_side != position_side:
                position_side = new_side
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
        
        signals[i] = desired_signal if position_side != 0 else 0.0
    
    return signals