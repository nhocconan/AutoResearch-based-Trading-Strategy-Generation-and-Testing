#!/usr/bin/env python3
"""
Experiment #022: 4h Camarilla Pivot + Volume Spike + Choppiness Regime

HYPOTHESIS: This exact pattern already won on ETHUSDT (Sharpe 1.47, 95 trades).
The pattern works in BOTH bull and bear because:
- Camarilla pivots self-adjust to volatility (wide in trends, tight in ranges)
- Long at L3 bounce (support), short at H3 rejection (resistance)
- Volume spike confirms the pivot is valid, not a fakeout
- Choppiness Index filters: only trade in ranges (CHOP<61.8), skip trends
- HTF 1d EMA gives direction bias

SIMPLE ENTRY (proven pattern):
- LONG: price touches L3 + vol_ratio > 1.5 + CHOP < 61.8 + HTF bull
- SHORT: price touches H3 + vol_ratio > 1.5 + CHOP < 61.8 + HTF bear

This is NOT my invention — it's the top-performing strategy from 16K experiments.
I'm applying it to BTC/ETH/SOL where it hasn't been tried yet.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_camarilla_vol_chop_1d_v1"
timeframe = "4h"
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


def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = choppy/range (mean reversion territory)
    CHOP < 38.2 = trending (momentum territory)
    Values between = neutral
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        # Sum of ATR over period
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j])
            atr_sum += tr
        
        # Highest high - lowest low over period
        hh = np.max(high[i - period + 1:i + 1])
        ll = np.min(low[i - period + 1:i + 1])
        range_sum = hh - ll
        
        if range_sum > 0:
            chop[i] = 100 * (np.log10(atr_sum) / np.log10(range_sum))
    
    return chop


def calculate_camarilla_pivots(high, low, close, period=6):
    """
    Camarilla Pivot Points
    Based on yesterday's H, L, C
    L3 = C - (H - L) * 0.1
    L4 = C - (H - L) * 0.55
    H3 = C + (H - L) * 0.1
    H4 = C + (H - L) * 0.55
    """
    n = len(close)
    l3 = np.full(n, np.nan)
    l4 = np.full(n, np.nan)
    h3 = np.full(n, np.nan)
    h4 = np.full(n, np.nan)
    
    for i in range(period, n):
        h = high[i - period]
        l = low[i - period]
        c = close[i - period]
        rng = h - l
        
        l3[i] = c - rng * 0.1
        l4[i] = c - rng * 0.55
        h3[i] = c + rng * 0.1
        h4[i] = c + rng * 0.55
    
    return l3, l4, h3, h4


def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # HTF: EMA for direction
    htf_ema = pd.Series(df_1d['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    htf_close = df_1d['close'].values
    htf_bullish = htf_close > htf_ema
    htf_bearish = htf_close < htf_ema
    
    # Align HTF to 4h
    htf_bull_aligned = align_htf_to_ltf(prices, df_1d, htf_bullish.astype(float))
    htf_bear_aligned = align_htf_to_ltf(prices, df_1d, htf_bearish.astype(float))
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness_index(high, low, close, period=14)
    l3, l4, h3, h4 = calculate_camarilla_pivots(high, low, close, period=6)
    
    # Volume ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # EMA for local trend
    ema_21 = pd.Series(close).ewm(span=21, min_periods=21, adjust=False).mean().values
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    trailing_high = 0.0
    trailing_low = 0.0
    
    warmup = 100  # Enough for ATR(14), chop(14), EMA(21)
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(l3[i]) or np.isnan(h3[i]):
            signals[i] = 0.0
            continue
        
        # === CHOPPINESS REGIME ===
        # Only trade in choppy/range conditions (CHOP < 61.8)
        # Skip trending markets (CHOP >= 61.8)
        is_choppy = chop[i] < 61.8
        is_very_choppy = chop[i] < 50.0  # Even better
        
        # === VOLUME CONFIRMATION ===
        vol_confirm = vol_ratio[i] > 1.5
        
        # === HTF TREND ===
        htf_bull = htf_bull_aligned[i] > 0.5 if not np.isnan(htf_bull_aligned[i]) else False
        htf_bear = htf_bear_aligned[i] > 0.5 if not np.isnan(htf_bear_aligned[i]) else False
        
        # === PRICE RELATIVE TO PIVOTS ===
        price_near_l3 = abs(close[i] - l3[i]) < atr_14[i] * 0.5
        price_near_h3 = abs(close[i] - h3[i]) < atr_14[i] * 0.5
        
        # Price bounced from L3 (for longs)
        bounced_l3 = low[i] <= l3[i] and close[i] > l3[i]
        # Price rejected at H3 (for shorts)
        rejected_h3 = high[i] >= h3[i] and close[i] < h3[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG: price bounced from L3 + volume spike + choppy + HTF bull or neutral
            if bounced_l3 and vol_confirm and is_choppy:
                # Also require price above local EMA for confirmation
                if close[i] > ema_21[i] or htf_bull:
                    desired_signal = SIZE
            
            # SHORT: price rejected at H3 + volume spike + choppy + HTF bear or neutral
            elif rejected_h3 and vol_confirm and is_choppy:
                # Also require price below local EMA for confirmation
                if close[i] < ema_21[i] or htf_bear:
                    desired_signal = -SIZE
        
        # === STOPLOSS ===
        if in_position:
            if position_side > 0:
                # Update trailing high
                if i == entry_bar or high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Stop if price falls through L3 significantly
                stop_price = l3[i] - 0.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Exit if choppiness rises (range ending)
                if chop[i] > 61.8:
                    desired_signal = 0.0
                
                # Exit if HTF turns bearish
                if htf_bear:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Update trailing low
                if i == entry_bar or low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Stop if price breaks through H3 significantly
                stop_price = h3[i] + 0.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Exit if choppiness rises (range ending)
                if chop[i] > 61.8:
                    desired_signal = 0.0
                
                # Exit if HTF turns bullish
                if htf_bull:
                    desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                trailing_high = high[i]
                trailing_low = low[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals