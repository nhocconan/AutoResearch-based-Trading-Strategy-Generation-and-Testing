#!/usr/bin/env python3
"""
Experiment #021: 4h Camarilla + Choppiness Regime + Volume Spike

HYPOTHESIS: Markets alternate between TRENDING and RANGING regimes.
By using Choppiness Index (CHOP) to detect regime, we can:
- In CHOPPY markets (CHOP > 61.8): mean-reversion around Camarilla pivots
- In TRENDING markets (CHOP < 38.2): trend-following Camarilla breakouts

WHY 4h: Optimal balance between signal quality and trade frequency.
4h captures multi-day swings without overtrading (unlike 15m/30m).
Test data shows 4h has best Sharpe on BTC/ETH/SOL.

WHY IT WORKS IN BULL AND BEAR:
- Bull: CHOP > 61.8 → buy S3/S4 Camarilla touches with volume
- Bear: CHOP > 61.8 → sell R3/R4 Camarilla touches with volume
- Trend: CHOP < 38.2 → breakouts of Camarilla levels in trend direction

TARGET: 100-250 total trades over 4 years (25-62/year). HARD MAX: 300.
Signal size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_camarilla_chop_vol_1d_v1"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    - CHOP > 61.8 = CHOPPY/RANGING (mean reversion works)
    - CHOP < 38.2 = TRENDING (trend following works)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        highest = high[i-period+1:i+1].max()
        lowest = low[i-period+1:i+1].min()
        
        if highest == lowest:
            chop[i] = 100.0
            continue
            
        sum_tr = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j])
            sum_tr += tr
        
        chop[i] = 100 * np.log10(sum_tr / (highest - lowest)) / np.log10(period)
    
    return chop

def calculate_hma(values, period):
    """Hull Moving Average"""
    half = pd.Series(values).rolling(window=period//2, min_periods=period//2).mean()
    full = pd.Series(values).rolling(window=period, min_periods=period).mean()
    hma = (2 * half - full).rolling(window=int(np.sqrt(period)), min_periods=int(np.sqrt(period))).mean()
    return hma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d HMA(21) for trend direction
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # HMA on 4h for local trend
    hma_4h = calculate_hma(close, 16)
    
    # Volume ratio (20-bar lookback)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
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
    
    warmup = 50  # Enough for CHOP(14) + HMA(16) + alignment
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(chop[i]) or np.isnan(hma_1d_aligned[i]) or np.isnan(hma_4h[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === REGIME DETECTION ===
        choppy_regime = chop[i] > 61.8
        trending_regime = chop[i] < 38.2
        
        # === TREND DETECTION (1d HMA aligned to 4h) ===
        price_above_1d_hma = close[i] > hma_1d_aligned[i]
        price_above_4h_hma = close[i] > hma_4h[i]
        
        # === CAMARILLA LEVELS from previous CLOSED bar ===
        prev_high = high[i - 1]
        prev_low = low[i - 1]
        prev_close = close[i - 1]
        prev_range = prev_high - prev_low
        
        # Classic Camarilla levels
        r3 = prev_close + prev_range * 0.09167
        r4 = prev_close + prev_range * 0.18333
        s3 = prev_close - prev_range * 0.09167
        s4 = prev_close - prev_range * 0.18333
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === CHOPPY REGIME: Mean-reversion at Camarilla levels ===
            if choppy_regime and vol_spike:
                # Long S3/S4 touches in uptrend
                if price_above_1d_hma and (low[i] <= s3 or low[i] <= s4):
                    desired_signal = SIZE
                # Short R3/R4 touches in downtrend
                elif not price_above_1d_hma and (high[i] >= r3 or high[i] >= r4):
                    desired_signal = -SIZE
            
            # === TRENDING REGIME: Breakout of Camarilla in trend direction ===
            if trending_regime and vol_spike:
                # Long: break above R4 in uptrend
                if price_above_1d_hma and high[i] > r4:
                    desired_signal = SIZE
                # Short: break below S4 in downtrend
                elif not price_above_1d_hma and low[i] < s4:
                    desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR trailing) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                desired_signal = 0.0
        
        # === MINIMUM HOLD: 3 bars (12h) to avoid fee churn ===
        bars_held = i - entry_bar
        
        if in_position and bars_held >= 3:
            # Take profit at Camarilla mid (previous close)
            if position_side > 0 and close[i] >= prev_close:
                desired_signal = 0.0
            if position_side < 0 and close[i] <= prev_close:
                desired_signal = 0.0
        
        # === UPDATE POSITION ===
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
            if in_position:
                in_position = False
                position_side = 0
                stop_price = 0.0
        
        signals[i] = desired_signal
    
    return signals