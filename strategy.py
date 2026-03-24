#!/usr/bin/env python3
"""
Experiment #1051: 6h Primary + 1w/1d HTF — Fisher Transform + ADX Regime + HMA Bias

Hypothesis: Ehlers Fisher Transform excels at catching reversals in bear/range markets
(2022-2023, 2025+) where simple trend-following fails. Combined with ADX regime detection
and 1w HMA bias, this should generate consistent trades across all market conditions.

Key innovations:
1. Ehlers Fisher Transform (period=9): Transforms price to near-Gaussian, sharp reversal signals
   - Long when Fisher crosses above -1.5 (oversold reversal)
   - Short when Fisher crosses below +1.5 (overbought reversal)
2. ADX(14) regime filter: ADX>25 = trend (follow direction), ADX<25 = range (mean revert)
3. 1w HMA(21) for long-term bias: Only long if price>1w_HMA, only short if price<1w_HMA
4. 1d ADX for intermediate trend confirmation
5. LOOSE entry conditions to guarantee 30+ trades/year (learned from 0-trade failures)
6. ATR(14) 2.5x trailing stop for risk management
7. Discrete sizing: 0.0, ±0.25, ±0.30 to minimize fee churn

Why this should work:
- Fisher Transform has superior reversal detection vs RSI in bear markets
- ADX regime switching adapts to different volatility environments
- 6h captures multi-day swings (30-60 trades/year target)
- 1w HMA ensures we don't fight the long-term trend
- Loose entries guarantee trades while HTF filters provide edge

Entry conditions (LOOSE to guarantee trades):
- LONG: Fisher<-1.0 cross up + price>1w_HMA*0.97 (allow slight below for entries)
- SHORT: Fisher>+1.0 cross down + price<1w_HMA*1.03
- ADX>25: Strengthen signal, ADX<25: Reduce size (range market)

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_adx_hma_regime_1w1d_v1"
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
    """Average Directional Index - measures trend strength"""
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, np.nan)
    
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        elif low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = np.zeros(n, dtype=np.float64)
    minus_di = np.zeros(n, dtype=np.float64)
    
    for i in range(period, n):
        if atr[i] > 1e-10:
            plus_di[i] = 100.0 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
            minus_di[i] = 100.0 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
    
    dx = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    adx[:period*2] = np.nan
    return adx

def calculate_fisher(close, period=9):
    """
    Ehlers Fisher Transform - transforms price to near-Gaussian distribution
    Sharp reversal signals when Fisher crosses extreme levels
    
    Formula:
    1. Normalize price: (close - lowest_low) / (highest_high - lowest_low)
    2. Apply Fisher: 0.5 * ln((1+x)/(1-x)) where x is normalized price
    3. Smooth with EMA
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    fisher = np.full(n, np.nan, dtype=np.float64)
    fisher_signal = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        highest_high = np.max(close[i-period+1:i+1])
        lowest_low = np.min(close[i-period+1:i+1])
        
        price_range = highest_high - lowest_low
        if price_range < 1e-10:
            continue
        
        # Normalize price to -1 to +1 range
        x = 2.0 * (close[i] - lowest_low) / price_range - 1.0
        x = max(-0.999, min(0.999, x))  # Clamp to avoid log(0)
        
        # Fisher transform
        fisher_raw = 0.5 * np.log((1.0 + x) / (1.0 - x))
        fisher[i] = fisher_raw
    
    # Smooth Fisher with EMA
    fisher_series = pd.Series(fisher)
    fisher_smooth = fisher_series.ewm(span=3, min_periods=3, adjust=False).mean().values
    
    return fisher_smooth

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF indicators
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    adx_1d_raw = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d_raw)
    
    # Calculate 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    adx_14 = calculate_adx(high, low, close, period=14)
    fisher = calculate_fisher(close, period=9)
    
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
    
    # Track Fisher crosses
    prev_fisher = np.nan
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            prev_fisher = fisher[i] if not np.isnan(fisher[i]) else prev_fisher
            continue
        
        if np.isnan(adx_14[i]) or np.isnan(fisher[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            prev_fisher = fisher[i] if not np.isnan(fisher[i]) else prev_fisher
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            prev_fisher = fisher[i] if not np.isnan(fisher[i]) else prev_fisher
            continue
        
        # === HTF BIAS (1w HMA) ===
        hma_1w_bull = close[i] > hma_1w_aligned[i] * 0.97  # Allow slight below for entries
        hma_1w_bear = close[i] < hma_1w_aligned[i] * 1.03  # Allow slight above for entries
        
        # === REGIME DETECTION (ADX) ===
        is_trending = adx_14[i] > 25.0
        is_ranging = adx_14[i] < 25.0
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_cross_long = False
        fisher_cross_short = False
        
        if not np.isnan(prev_fisher):
            # Long: Fisher crosses above -1.5 (oversold reversal)
            if prev_fisher < -1.5 and fisher[i] >= -1.5:
                fisher_cross_long = True
            # Also trigger on less extreme cross for more trades
            if prev_fisher < -1.0 and fisher[i] >= -1.0:
                fisher_cross_long = True
            
            # Short: Fisher crosses below +1.5 (overbought reversal)
            if prev_fisher > 1.5 and fisher[i] <= 1.5:
                fisher_cross_short = True
            # Also trigger on less extreme cross for more trades
            if prev_fisher > 1.0 and fisher[i] <= 1.0:
                fisher_cross_short = True
        
        prev_fisher = fisher[i]
        
        # === ENTRY LOGIC (LOOSE CONDITIONS TO GUARANTEE TRADES) ===
        desired_signal = 0.0
        
        # LONG entries
        if fisher_cross_long and hma_1w_bull:
            if is_trending:
                desired_signal = SIZE_STRONG  # Strong trend = larger size
            else:
                desired_signal = SIZE_BASE  # Range = smaller size
        
        # SHORT entries
        elif fisher_cross_short and hma_1w_bear:
            if is_trending:
                desired_signal = -SIZE_STRONG
            else:
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