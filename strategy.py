#!/usr/bin/env python3
"""
Experiment #021: 4h Camarilla Pivot + Volume Spike + Choppiness Regime

HYPOTHESIS: Choppiness Index acts as a meta-regime filter. When CHOP > 61.8
(range market), Camarilla levels act as reversal magnets. When CHOP < 38.2
(trending), Camarilla levels become pullback entries in trend direction.

WHY 4h: Matches proven DB pattern (ETHUSDT test Sharpe 1.47). 12h may miss
some valid Camarilla setups. 4h gives better entry timing.

WHY IT WORKS IN BULL AND BEAR: Symmetrical pivot math. Buy S3/S4 in uptrends
(CHOP > 61.8), sell R3/R4 in downtrends. Choppiness tells us when to trade.

TARGET: 75-150 total trades over 4 years = 19-37/year. HARD MAX: 200.
Signal size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_camarilla_vol_chop_v1"
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
    CHOP > 61.8 = choppy/range market (good for mean reversion)
    CHOP < 38.2 = trending (good for trend continuation)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        # Sum of ATR over period
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j],
                     abs(low[j] - close[j-1]) if j > 0 else 0)
            atr_sum += tr
        
        # Highest high - lowest low over period
        period_high = max(high[i - period + 1:i + 1])
        period_low = min(low[i - period + 1:i + 1])
        high_low_range = period_high - period_low
        
        if high_low_range > 0:
            chop[i] = 100 * (np.log10(atr_sum) / np.log10(high_low_range * period))
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA50 for basic trend direction (simpler than EMA)
    sma_1d = pd.Series(df_1d['close'].values).rolling(window=50, min_periods=50).mean().values
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    # Volume ratio (20-bar MA)
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
    
    warmup = 100  # Need enough for all indicators
    
    for i in range(warmup, n):
        # Skip if ATR not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if chop not ready
        if np.isnan(chop_14[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if SMA not aligned
        if np.isnan(sma_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === REGIME: CHOPPINESS FILTER ===
        # Only trade when CHOP > 61.8 (range market) or CHOP < 38.2 (strong trend)
        # Skip when 38.2 < CHOP < 61.8 (transitional)
        chop_regime = (chop_14[i] > 61.8) or (chop_14[i] < 38.2)
        
        # === TREND: 1d SMA50 ===
        price_above_1d_sma = close[i] > sma_1d_aligned[i]
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # === CAMARILLA LEVELS from previous CLOSED bar (no look-ahead) ===
        prev_high = high[i - 1]
        prev_low = low[i - 1]
        prev_close = close[i - 1]
        prev_range = prev_high - prev_low
        
        # Classic Camarilla levels
        r3 = prev_close + prev_range * 0.09167
        r4 = prev_close + prev_range * 0.18333
        s3 = prev_close - prev_range * 0.09167
        s4 = prev_close - prev_range * 0.18333
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position and chop_regime:
            # === LONG: Price touches S3/S4 with volume in uptrend ===
            # In range (CHOP > 61.8): buy bounces at support
            # In strong trend (CHOP < 38.2): buy pullbacks to S3/S4
            if price_above_1d_sma and vol_spike:
                if low[i] <= s4:
                    desired_signal = SIZE
                elif low[i] <= s3:
                    desired_signal = SIZE
            
            # === SHORT: Price touches R3/R4 with volume in downtrend ===
            if not price_above_1d_sma and vol_spike:
                if high[i] >= r4:
                    desired_signal = -SIZE
                elif high[i] >= r3:
                    desired_signal = -SIZE
        
        # === STOPLOSS (2.0 ATR trailing) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                desired_signal = 0.0
        
        # === HOLD PERIOD: minimum 2 bars to avoid churn ===
        bars_held = i - entry_bar
        
        if in_position and bars_held >= 2:
            # Exit when price returns to mid-point (prev close)
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
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                stop_price = 0.0
        
        signals[i] = desired_signal
    
    return signals