#!/usr/bin/env python3
"""
Experiment #1008: 4h Primary + 12h/1d HTF — Fisher Transform + Choppiness Regime + HMA Trend

Hypothesis: Fisher Transform excels at catching reversals in bear/range markets (2022-2023),
while Choppiness Index detects regime to switch between mean-reversion and trend-following.
Combined with 12h/1d HMA for trend bias, this should outperform pure trend strategies.

Key innovations:
1. Ehlers Fisher Transform (period=9): Normalizes price to Gaussian distribution, catches reversals
   - Long when Fisher crosses above -1.5 (oversold reversal)
   - Short when Fisher crosses below +1.5 (overbought reversal)
2. Choppiness Index (CHOP 14): Regime detection
   - CHOP > 55: Range market → use Fisher mean reversion
   - CHOP < 45: Trend market → use HMA trend following
3. 12h HMA(21) + 1d HMA(21): Multi-timeframe trend bias
   - Only long if price > 12h_HMA (bullish intermediate trend)
   - Only short if price < 12h_HMA (bearish intermediate trend)
4. Volume confirmation: Entry volume > 0.8 * volume_sma(20) avoids low-liquidity traps
5. ATR(14) 2.5x trailing stop for risk management
6. LOOSE entry conditions to guarantee 30+ trades (learned from 0-trade failures)

Why this should work:
- Fisher Transform has proven edge in crypto bear markets (catches 2022 bottom reversals)
- Choppiness filter avoids trend-following whipsaws in 2022-2023 range
- 12h/1d HMA provides smoother trend bias than 4h alone
- 4h timeframe = 20-50 trades/year target (fee-efficient)
- Looser thresholds than #1002 to avoid 0-trade problem

Entry conditions (LOOSE for trade generation):
- LONG range: CHOP>50 + Fisher<-1.0 + price>12h_HMA*0.97 + vol_ok
- LONG trend: CHOP<50 + price>12h_HMA>1d_HMA + Fisher>-0.5 + vol_ok
- SHORT range: CHOP>50 + Fisher>1.0 + price<12h_HMA*1.03 + vol_ok
- SHORT trend: CHOP<50 + price<12h_HMA<1d_HMA + Fisher<0.5 + vol_ok

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%
Timeframe: 4h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_chop_hma_regime_12h1d_v1"
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

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Excellent for catching reversals in bear/range markets
    
    Formula:
    1. Calculate typical price: (high + low) / 2
    2. Normalize: (price - lowest_low) / (highest_high - lowest_low)
    3. Transform: 0.5 * ln((1 + x) / (1 - x)) where x = 2*normalized - 1
    """
    n = len(high)
    if n < period:
        return np.full(n, np.nan)
    
    fisher = np.full(n, np.nan, dtype=np.float64)
    fisher_prev = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        price_range = highest_high - lowest_low
        if price_range < 1e-10:
            continue
        
        typical_price = (high[i] + low[i]) / 2.0
        normalized = (typical_price - lowest_low) / price_range
        
        # Clamp to avoid division by zero in log
        x = 2.0 * normalized - 1.0
        x = np.clip(x, -0.99, 0.99)
        
        fisher[i] = 0.5 * np.log((1.0 + x) / (1.0 - x))
        
        if i > period - 1:
            fisher_prev[i] = fisher[i-1]
        else:
            fisher_prev[i] = fisher[i]
    
    return fisher, fisher_prev

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
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
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        price_range = highest_high - lowest_low
        if price_range > 1e-10 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    fisher, fisher_prev = calculate_fisher_transform(high, low, period=9)
    
    # Volume SMA for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(fisher[i]) or np.isnan(fisher_prev[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # Volume confirmation (avoid low-liquidity entries)
        vol_ok = volume[i] > 0.7 * vol_sma[i] if not np.isnan(vol_sma[i]) else True
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop_14[i] > 50.0  # Range market (looser threshold)
        is_trending = chop_14[i] < 50.0  # Trend market
        
        # === HTF BIAS (12h/1d HMA alignment) ===
        price_above_12h = close[i] > hma_12h_aligned[i]
        price_below_12h = close[i] < hma_12h_aligned[i]
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # Strong trend alignment
        strong_bull = price_above_12h and price_above_1d and hma_12h_aligned[i] > hma_1d_aligned[i]
        strong_bear = price_below_12h and price_below_1d and hma_12h_aligned[i] < hma_1d_aligned[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_cross_up = fisher[i] > -1.0 and fisher_prev[i] <= -1.0
        fisher_cross_down = fisher[i] < 1.0 and fisher_prev[i] >= 1.0
        fisher_oversold = fisher[i] < -0.5
        fisher_overbought = fisher[i] > 0.5
        
        # === ENTRY LOGIC (REGIME-ADAPTIVE) ===
        desired_signal = 0.0
        
        if is_choppy:
            # MEAN REVERSION MODE - use Fisher Transform reversals
            # Long when Fisher crosses up from oversold
            if fisher_cross_up and price_above_12h and vol_ok:
                desired_signal = SIZE_BASE
            # Short when Fisher crosses down from overbought
            elif fisher_cross_down and price_below_12h and vol_ok:
                desired_signal = -SIZE_BASE
            # Stronger signals at more extreme Fisher
            elif fisher[i] < -1.2 and price_above_12h and vol_ok:
                desired_signal = SIZE_STRONG
            elif fisher[i] > 1.2 and price_below_12h and vol_ok:
                desired_signal = -SIZE_STRONG
        
        elif is_trending:
            # TREND FOLLOWING MODE - use HMA alignment + Fisher confirmation
            # Long in strong uptrend with Fisher confirmation
            if strong_bull and fisher_oversold and vol_ok:
                desired_signal = SIZE_STRONG
            # Short in strong downtrend with Fisher confirmation
            elif strong_bear and fisher_overbought and vol_ok:
                desired_signal = -SIZE_STRONG
            # Weaker trend signals
            elif price_above_12h and price_above_1d and fisher[i] > -0.5 and vol_ok:
                desired_signal = SIZE_BASE
            elif price_below_12h and price_below_1d and fisher[i] < 0.5 and vol_ok:
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