#!/usr/bin/env python3
"""
Experiment #1291: 6h Primary + 1w/1d HTF — Vol Spike Mean Reversion + HTF Trend Filter

Hypothesis: Current best 6h strategy (KAMA+ROC) achieved Sharpe=0.447 but is pure trend-following.
This variant exploits vol-spike mean reversion (proven in 2022 crash) while respecting HTF trend.

Key innovations:
1. Vol spike detection: ATR(7)/ATR(30) > 2.0 signals panic/extreme conditions
2. BB(20, 2.5) extremes for mean reversion entry (wider than standard 2.0)
3. 1w HMA(21) for major regime bias (only long if weekly bullish)
4. 1d HMA(21) for intermediate trend confirmation
5. ADX(14) regime switch: >25 = trend follow, <20 = mean revert
6. Asymmetric sizing: 0.30 for high-conviction, 0.20 for standard

Why this should beat KAMA+ROC:
- Vol spike reversion worked through 2022 crash (when trend strategies failed)
- Dual HTF filter prevents counter-trend mean reversion disasters
- ADX regime adaptation = right tool for right market condition
- Wider BB (2.5 std) = fewer but higher quality mean reversion trades
- 6h timeframe naturally produces 30-60 trades/year

Entry logic:
- TREND MODE (ADX>25): Long if 1w/1d HMA bullish + 6h pullback to HMA(21)
- MEAN REVERT MODE (ADX<20): Long if vol spike + price < BB_lower(2.5) + 1w HMA bullish
- SHORT: Mirror logic for bearish conditions

Target: Sharpe>0.5, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 6h
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_vol_spike_mean_reversion_htf_trend_1w1d_v1"
timeframe = "6h"
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

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength"""
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, np.nan)
    
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    plus_di = np.full(n, np.nan, dtype=np.float64)
    minus_di = np.full(n, np.nan, dtype=np.float64)
    
    plus_tr_sum = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_sum = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_sum = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    for i in range(period, n):
        if plus_tr_sum[i] > 1e-10:
            plus_di[i] = 100.0 * plus_dm_sum[i] / plus_tr_sum[i]
            minus_di[i] = 100.0 * minus_dm_sum[i] / plus_tr_sum[i]
    
    dx = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if plus_di[i] + minus_di[i] > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

def calculate_bollinger_bands(close, period=20, std_dev=2.5):
    """Bollinger Bands with configurable std dev"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    
    return upper, sma, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    adx_14 = calculate_adx(high, low, close, period=14)
    hma_6h = calculate_hma(close, period=21)
    
    bb_upper, bb_mid, bb_lower = calculate_bollinger_bands(close, period=20, std_dev=2.5)
    
    # Vol spike ratio: ATR(7) / ATR(30)
    vol_spike_ratio = np.full(n, np.nan, dtype=np.float64)
    for i in range(30, n):
        if atr_30[i] > 1e-10 and not np.isnan(atr_7[i]):
            vol_spike_ratio[i] = atr_7[i] / atr_30[i]
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup period
    min_bars = 150
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(adx_14[i]) or np.isnan(hma_6h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1w_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(bb_lower[i]) or np.isnan(bb_upper[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF TREND BIAS ===
        # Weekly trend: price above/below 1w HMA
        weekly_bullish = close[i] > hma_1w_aligned[i]
        weekly_bearish = close[i] < hma_1w_aligned[i]
        
        # Daily trend: price above/below 1d HMA
        daily_bullish = close[i] > hma_1d_aligned[i]
        daily_bearish = close[i] < hma_1d_aligned[i]
        
        # 1w HMA slope (compare to 2 bars ago for stability)
        hma_1w_slope = 0.0
        if i >= 2 and not np.isnan(hma_1w_aligned[i-2]):
            hma_1w_slope = hma_1w_aligned[i] - hma_1w_aligned[i-2]
        
        hma_1d_slope = 0.0
        if i >= 2 and not np.isnan(hma_1d_aligned[i-2]):
            hma_1d_slope = hma_1d_aligned[i] - hma_1d_aligned[i-2]
        
        # === REGIME DETECTION (ADX) ===
        adx = adx_14[i]
        is_trending = adx > 25.0
        is_ranging = adx < 20.0
        
        # === VOL SPIKE DETECTION ===
        vol_ratio = vol_spike_ratio[i]
        is_vol_spike = vol_ratio > 2.0 if not np.isnan(vol_ratio) else False
        
        # === 6H LOCAL TREND ===
        price_above_6h = close[i] > hma_6h[i]
        price_below_6h = close[i] < hma_6h[i]
        
        # === BB POSITION ===
        bb_position = (close[i] - bb_mid[i]) / (bb_upper[i] - bb_lower[i]) * 2.0 if (bb_upper[i] - bb_lower[i]) > 1e-10 else 0.0
        near_bb_lower = close[i] < bb_lower[i] * 1.005  # Within 0.5% of lower band
        near_bb_upper = close[i] > bb_upper[i] * 0.995  # Within 0.5% of upper band
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG ENTRIES
        if weekly_bullish and hma_1w_slope >= 0:
            if is_trending:
                # Trend mode: pullback to 6h HMA with daily confirmation
                if daily_bullish and price_below_6h and close[i] > hma_6h[i] * 0.98:
                    if adx > 22.0:  # Slightly lower threshold for entry
                        desired_signal = SIZE_BASE
            elif is_ranging:
                # Mean reversion mode: vol spike + BB lower
                if is_vol_spike and near_bb_lower:
                    desired_signal = SIZE_STRONG  # High conviction on vol spike reversal
                elif near_bb_lower and vol_ratio > 1.5:
                    desired_signal = SIZE_BASE
        
        # SHORT ENTRIES
        if weekly_bearish and hma_1w_slope <= 0:
            if is_trending:
                # Trend mode: rally to 6h HMA with daily confirmation
                if daily_bearish and price_above_6h and close[i] < hma_6h[i] * 1.02:
                    if adx > 22.0:
                        desired_signal = -SIZE_BASE
            elif is_ranging:
                # Mean reversion mode: vol spike + BB upper
                if is_vol_spike and near_bb_upper:
                    desired_signal = -SIZE_STRONG  # High conviction on vol spike reversal
                elif near_bb_upper and vol_ratio > 1.5:
                    desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
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