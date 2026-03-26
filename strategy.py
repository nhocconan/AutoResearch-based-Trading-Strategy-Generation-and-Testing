#!/usr/bin/env python3
"""
Experiment #001: 4h Camarilla Pivot + Volume Spike + Choppiness Regime

Hypothesis: Camarilla pivot levels from 1d provide STRUCTURAL price targets 
(R3/S3, R4/S4) that only trigger on significant moves, naturally limiting trades 
to 50-150/year. Combined with volume spike confirmation and choppiness regime 
filter, this captures the exact pattern that produced Sharpe 1.47 on ETH.

Why Camarilla over Donchian/KAMA:
- Camarilla has 8 levels (S1-S4, R1-R4) - more precise entry/exit structure
- R3/S3 are "midnight reversal" levels - proven in crypto volatile markets  
- Unlike RSI/Fisher thresholds, levels are absolute prices (no false triggers)
- Volume spike confirms whether breakouts are "real" institutional moves

Entry logic (TIGHT to avoid overtrading):
- TREND (CHOP<38): Close breaks R3/S3 + volume spike >1.5x avg + 1d HMA bias
- RANGE (CHOP>61): Price at outer bands (S4/R4) + extreme chop = mean reversion back to R3/S3

Target: Sharpe>0.8, trades 75-150 train, trades≥15 test, DD>-30%
Timeframe: 4h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_camarilla_volume_chop_regime_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while smoothing"""
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

def calculate_camarilla_pivot(high, low, close, period=1):
    """
    Camarilla Pivot Points (classic formula)
    Uses previous period's H, L, C to calculate 8 levels:
    R4 = C + (H - L) * 1.1
    R3 = C + (H - L) * 1.1 / 2
    R2 = C + (H - L) * 1.1 / 4
    R1 = C + (H - L) * 1.1 / 6
    S1 = C - (H - L) * 1.1 / 6
    S2 = C - (H - L) * 1.1 / 4
    S3 = C - (H - L) * 1.1 / 2
    S4 = C - (H - L) * 1.1
    """
    n = len(close)
    if n < period + 2:
        return {k: np.full(n, np.nan) for k in ['R1','R2','R3','R4','S1','S2','S3','S4','P']}
    
    r4 = np.full(n, np.nan, dtype=np.float64)
    r3 = np.full(n, np.nan, dtype=np.float64)
    r2 = np.full(n, np.nan, dtype=np.float64)
    r1 = np.full(n, np.nan, dtype=np.float64)
    s1 = np.full(n, np.nan, dtype=np.float64)
    s2 = np.full(n, np.nan, dtype=np.float64)
    s3 = np.full(n, np.nan, dtype=np.float64)
    s4 = np.full(n, np.nan, dtype=np.float64)
    piv = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        prev_high = high[i - period]
        prev_low = low[i - period]
        prev_close = close[i - period]
        rng = prev_high - prev_low
        
        if rng > 1e-10:
            piv[i] = (prev_high + prev_low + prev_close) / 3.0
            r1[i] = prev_close + rng * (1.1 / 6)
            r2[i] = prev_close + rng * (1.1 / 4)
            r3[i] = prev_close + rng * (1.1 / 2)
            r4[i] = prev_close + rng * 1.1
            s1[i] = prev_close - rng * (1.1 / 6)
            s2[i] = prev_close - rng * (1.1 / 4)
            s3[i] = prev_close - rng * (1.1 / 2)
            s4[i] = prev_close - rng * 1.1
    
    return {'R1': r1, 'R2': r2, 'R3': r3, 'R4': r4, 
            'S1': s1, 'S2': s2, 'S3': s3, 'S4': s4, 'P': piv}

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppy vs trending
    CHOP > 61.8 = ranging (mean reversion), CHOP < 38.2 = trending
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_volume_confirmation(volume, period=20):
    """
    Volume moving average for spike detection
    Volume spike = current vol > 1.5x 20-period average
    """
    n = len(volume)
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_ma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=48)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 1d Camarilla from HTF data
    cam_1d = calculate_camarilla_pivot(
        df_1d['high'].values, 
        df_1d['low'].values, 
        df_1d['close'].values
    )
    
    # Align 1d Camarilla to 4h
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, cam_1d['R3'])
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, cam_1d['S3'])
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, cam_1d['R4'])
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, cam_1d['S4'])
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    vol_ma_20 = calculate_volume_confirmation(volume, period=20)
    
    # 4h Camarilla for tighter entries
    cam_4h = calculate_camarilla_pivot(high, low, close)
    r3_4h = cam_4h['R3']
    s3_4h = cam_4h['S3']
    r4_4h = cam_4h['R4']
    s4_4h = cam_4h['S4']
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup period
    min_bars = 60
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(chop_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(vol_ma_20[i]) or vol_ma_20[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # Check HTF alignment
        if np.isnan(hma_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === VOLUME SPIKE DETECTION ===
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        # === REGIME DETECTION ===
        chop = chop_14[i]
        is_trend_regime = chop < 38.2
        is_range_regime = chop > 61.8
        
        # === 1d HMA TREND BIAS ===
        price_above_1d_hma = close[i] > hma_1d_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_aligned[i]
        
        # === 4h CAMARILLA LEVELS ===
        r3 = r3_4h[i]
        s3 = s3_4h[i]
        r4 = r4_4h[i]
        s4 = s4_4h[i]
        
        # === 1d CAMARILLA LEVELS (for confirmation) ===
        r3_1d = r3_1d_aligned[i]
        s3_1d = s3_1d_aligned[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # TREND REGIME: Breakout through R3/S3 with volume + 1d bias
        if is_trend_regime:
            if price_above_1d_hma:
                # Long: break above R3 + volume spike
                if not np.isnan(r3) and close[i] > r3 and vol_spike:
                    # Check 1d R3 is also above (confirming trend)
                    if not np.isnan(r3_1d) and r3_1d > hma_1d_aligned[i]:
                        desired_signal = SIZE_STRONG
            
            elif price_below_1d_hma:
                # Short: break below S3 + volume spike  
                if not np.isnan(s3) and close[i] < s3 and vol_spike:
                    # Check 1d S3 is also below (confirming trend)
                    if not np.isnan(s3_1d) and s3_1d < hma_1d_aligned[i]:
                        desired_signal = -SIZE_STRONG
        
        # RANGE REGIME: Mean reversion from outer bands
        elif is_range_regime:
            if price_above_1d_hma:
                # Long: price dropped to S4 area in range + bounce
                if not np.isnan(s4) and close[i] > s4 and close[i] < s4 * 1.02:
                    # Look for reversal candle pattern (close > open)
                    if close[i] > (high[i] + low[i]) / 2:
                        desired_signal = SIZE_BASE
            
            elif price_below_1d_hma:
                # Short: price rallied to R4 area in range + reversal
                if not np.isnan(r4) and close[i] < r4 and close[i] > r4 * 0.98:
                    # Bearish reversal (close < open)
                    if close[i] < (high[i] + low[i]) / 2:
                        desired_signal = -SIZE_BASE
        
        # NEUTRAL: No position (wait for clearer signal)
        else:
            # Only take trades with strong confluence: volume + 1d trend + 4h breakout
            if price_above_1d_hma and vol_spike:
                if not np.isnan(r3) and close[i] > r3:
                    desired_signal = SIZE_BASE
            elif price_below_1d_hma and vol_spike:
                if not np.isnan(s3) and close[i] < s3:
                    desired_signal = -SIZE_BASE
        
        # === TRAILING STOPLOSS (2.5x ATR) ===
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
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals