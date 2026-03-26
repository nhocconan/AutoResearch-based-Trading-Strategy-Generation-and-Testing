#!/usr/bin/env python3
"""
Experiment #012: 12h Camarilla Pivot Breakout + Volume Spike + 1d Trend

HYPOTHESIS: Camarilla pivots (R3/S3/R4/S4) mark institutional supply/demand zones.
When price breaks R3 with volume spike AND aligns with 1d HMA trend, it's a 
high-probability continuation trade. In bear markets (price < 1d HMA), short 
rallies that fail at R3 provide excellent shorts. 12h reduces churn vs 4h.

WHY 12h: Slower timeframe = fewer trades = less fee drag. Camarilla on 12h 
captures multi-day swings without noise of lower TFs.

KEY INSIGHT FROM DB: gen_camarilla_pivot_volume_spike_choppiness_4h_v1 achieved 
test Sharpe 1.471 on ETH. Adapting to 12h for even fewer trades.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_camarilla_vol_1d_hma_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

def calculate_camarilla_pivots(high, low, close):
    """Camarilla pivot levels - returns R3, R4, S3, S4, P"""
    n = len(close)
    r3 = np.full(n, np.nan, dtype=np.float64)
    r4 = np.full(n, np.nan, dtype=np.float64)
    s3 = np.full(n, np.nan, dtype=np.float64)
    s4 = np.full(n, np.nan, dtype=np.float64)
    piv = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(1, n):
        if np.isnan(high[i]) or np.isnan(low[i]) or np.isnan(close[i]):
            continue
        h = high[i]
        l = low[i]
        c = close[i]
        rng = h - l
        
        piv[i] = (h + l + c) / 3.0
        r3 = np.where(np.arange(n) == i, c + rng * 1.1, r3)
        r4 = np.where(np.arange(n) == i, c + rng * 1.2, r4)
        s3 = np.where(np.arange(n) == i, c - rng * 1.1, s3)
        s4 = np.where(np.arange(n) == i, c - rng * 1.2, s4)
    
    return r3, r4, s3, s4, piv

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
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Local indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume MA
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
    
    # Camarilla pivots (vectorized)
    pivots_r3 = np.full(n, np.nan, dtype=np.float64)
    pivots_r4 = np.full(n, np.nan, dtype=np.float64)
    pivots_s3 = np.full(n, np.nan, dtype=np.float64)
    pivots_s4 = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(1, n):
        rng = high[i] - low[i]
        c = close[i]
        pivots_r3[i] = c + rng * 1.1
        pivots_r4[i] = c + rng * 1.2
        pivots_s3[i] = c - rng * 1.1
        pivots_s4[i] = c - rng * 1.2
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(pivots_r3[i]) or np.isnan(pivots_s3[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        r3 = pivots_r3[i]
        r4 = pivots_r4[i]
        s3 = pivots_s3[i]
        s4 = pivots_s4[i]
        
        # === TREND BIAS (1d HMA) ===
        price_above_1d_hma = close[i] > hma_1d_aligned[i]
        trend_bullish = price_above_1d_hma
        trend_bearish = not price_above_1d_hma
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === RSI FOR MOMENTUM ===
        rsi_val = rsi[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Break above R3 with volume + bullish trend ===
            # Price breaks R3 AND closes above R3 AND volume confirms
            if close[i] > r3 and vol_spike and trend_bullish:
                desired_signal = SIZE
            
            # === SHORT: Break below S3 with volume + bearish trend ===
            if close[i] < s3 and vol_spike and trend_bearish:
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR) ===
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
        
        # === EXIT: Take profit at opposite pivot OR RSI extreme ===
        take_profit = False
        
        if in_position and position_side > 0:
            # TP: price reaches R4 (aggressive target)
            if high[i] >= r4:
                take_profit = True
            # OR: trend flips bearish
            if trend_bearish and rsi_val > 65:
                take_profit = True
        
        if in_position and position_side < 0:
            # TP: price reaches S4
            if low[i] <= s4:
                take_profit = True
            # OR: trend flips bullish
            if trend_bullish and rsi_val < 35:
                take_profit = True
        
        if take_profit:
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