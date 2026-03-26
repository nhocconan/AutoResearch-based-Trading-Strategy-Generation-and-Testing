#!/usr/bin/env python3
"""
Experiment #011: 6h Camarilla Pivot Breakout + ATR Volatility Expansion + 1d Choppiness Regime

HYPOTHESIS: Camarilla pivot levels (R3/S3) act as key institutional support/resistance.
In trending markets, price breaks through R3/S3 with ATR expansion = strong momentum.
The 1d Choppiness Index filters ranging markets where breakouts fail.
This captures major moves while avoiding the 2022 whipsaw at the bottom.

TIMEFRAME: 6h primary
HTF: 1d for Choppiness Index regime
TARGET: 75-200 total trades over 4 years (19-50/year)
SIZE: 0.25
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_camarilla_vol_expansion_chop_1d_v1"
timeframe = "6h"
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

def calculate_camarilla_levels(high, low, close, lookback=1):
    """Calculate Camarilla pivot levels from previous period.
    Returns arrays aligned with input index.
    """
    n = len(close)
    if n < lookback + 1:
        return (np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan),
                np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan),
                np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan))
    
    r1 = np.full(n, np.nan, dtype=np.float64)
    s1 = np.full(n, np.nan, dtype=np.float64)
    r2 = np.full(n, np.nan, dtype=np.float64)
    s2 = np.full(n, np.nan, dtype=np.float64)
    r3 = np.full(n, np.nan, dtype=np.float64)
    s3 = np.full(n, np.nan, dtype=np.float64)
    r4 = np.full(n, np.nan, dtype=np.float64)
    s4 = np.full(n, np.nan, dtype=np.float64)
    pivot = np.full(n, np.nan, dtype=np.float64)
    
    # Need lookback bars to calculate first level
    for i in range(lookback, n):
        h = high[i - lookback]
        l = low[i - lookback]
        c = close[i - lookback]
        
        rng = h - l
        
        pivot[i] = (h + l + c) / 3.0
        r1[i] = c + rng * (1.1 / 12.0)
        s1[i] = c - rng * (1.1 / 12.0)
        r2[i] = c + rng * (1.1 / 6.0)
        s2[i] = c - rng * (1.1 / 6.0)
        r3[i] = c + rng * (1.1 / 4.0)
        s3[i] = c - rng * (1.1 / 4.0)
        r4[i] = c + rng * (1.1 / 2.0)
        s4[i] = c - rng * (1.1 / 2.0)
    
    return r1, s1, r2, s2, r3, s3, r4, s4, pivot

def calculate_choppiness_index(high, low, close, period=14):
    """Choppiness Index - measures market choppiness vs trending.
    CHOP < 38.2 = trending (good for trend following)
    CHOP > 61.8 = ranging (mean reversion)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        # Sum of ATR over period
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr1 = high[j] - low[j]
            tr2 = abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j]
            tr3 = abs(low[j] - close[j-1]) if j > 0 else high[j] - low[j]
            atr_sum += max(tr1, tr2, tr3)
        
        # Highest high - lowest low over period
        hh = np.max(high[i - period + 1:i + 1])
        ll = np.min(low[i - period + 1:i + 1])
        
        if hh - ll > 0:
            chop[i] = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d Choppiness Index - calculate from HTF data
    chop_1d_raw = calculate_choppiness_index(
        df_1d['high'].values, 
        df_1d['low'].values, 
        df_1d['close'].values, 
        period=14
    )
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d_raw)
    
    # 1d SMA for trend direction
    sma_1d_raw = pd.Series(df_1d['close'].values).rolling(window=50, min_periods=50).mean().values
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d_raw)
    
    # Calculate local 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_30 = calculate_atr(high, low, close, period=30)
    
    # ATR ratio for volatility expansion detection
    atr_ratio = atr_14 / np.where(atr_30 > 0, atr_30, 1)
    
    # Camarilla levels (use 1 bar lookback = previous 6h bar)
    r3, s3, r4, s4, _, _, _, _, _ = calculate_camarilla_levels(high, low, close, lookback=1)
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # RSI for momentum
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, min_periods=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = (100 - (100 / (1 + rs))).values
    
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(r3[i]) or np.isnan(s3[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(chop_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME CHECK (1d Choppiness) ===
        chop_val = chop_1d_aligned[i]
        is_trending = chop_val < 50.0  # Less choppy = trending
        is_ranging = chop_val > 58.0  # Very choppy = range
        
        # === TREND DIRECTION (1d SMA) ===
        price_above_sma = close[i] > sma_1d_aligned[i] if not np.isnan(sma_1d_aligned[i]) else True
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.4
        
        # === VOLATILITY EXPANSION ===
        vol_expansion = atr_ratio[i] > 1.15  # Today's ATR higher than recent average
        
        # === RSI FOR MOMENTUM ===
        rsi_val = rsi[i]
        
        # === CAMARILLA LEVELS ===
        r3_val = r3[i]
        s3_val = s3[i]
        r4_val = r4[i]
        s4_val = s4[i]
        
        # === ENTRY LOGIC ===
        # Breakout up through R3 + vol expansion + volume = strong momentum
        # In trending regime only (avoid whipsaw in range)
        
        desired_signal = 0.0
        
        if not in_position:
            # === NEW LONG ENTRY ===
            # Price breaks above R3 with vol expansion + volume
            if close[i] > r3_val and vol_expansion and vol_spike:
                # Only in trending regime or with bullish 1d trend
                if is_trending or price_above_sma:
                    desired_signal = SIZE
            
            # === NEW SHORT ENTRY ===
            # Price breaks below S3 with vol expansion + volume
            if close[i] < s3_val and vol_expansion and vol_spike:
                # Only in trending regime or with bearish 1d trend
                if is_trending or not price_above_sma:
                    desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === EXIT CONDITIONS ===
        exit_triggered = False
        
        if in_position and position_side > 0:
            # Exit long: price falls below S3 OR RSI < 35 OR vol contraction
            if close[i] < s3_val:
                exit_triggered = True
            if rsi_val < 35:
                exit_triggered = True
            # Extreme overbought - take profit
            if rsi_val > 80:
                exit_triggered = True
        
        if in_position and position_side < 0:
            # Exit short: price rises above R3 OR RSI > 65 OR vol contraction
            if close[i] > r3_val:
                exit_triggered = True
            if rsi_val > 65:
                exit_triggered = True
            # Extreme oversold - take profit
            if rsi_val < 20:
                exit_triggered = True
        
        if exit_triggered:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                entry_bar = i
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
            else:
                # Same direction - maintain position
                pass
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                entry_bar = 0
        
        signals[i] = desired_signal
    
    return signals