#!/usr/bin/env python3
"""
Experiment #1027: 6h Primary + 1d/1w HTF — Ehlers Fisher + HMA Trend + Choppiness Regime

Hypothesis: Combining Ehlers Fisher Transform (proven reversal catcher in bear markets) with
HMA trend filtering and Choppiness Index regime detection will outperform pure trend or pure
mean-reversion strategies on 6h timeframe.

Key innovations:
1. Ehlers Fisher Transform (period=9): Normalizes price to Gaussian distribution
   - Long when Fisher crosses above -1.5 (oversold reversal)
   - Short when Fisher crosses below +1.5 (overbought reversal)
2. Choppiness Index regime filter: CHOP>61.8 = range (use Fisher extremes), CHOP<38.2 = trend (use HMA direction)
3. 1d/1w HMA(21) alignment for long-term bias (only trade with HTF trend)
4. ATR(14) 2.5x trailing stop for risk management
5. Discrete sizing: 0.0, ±0.25, ±0.30 to minimize fee churn

Why this should work:
- Fisher Transform catches reversals in bear rallies (2022-2023 range markets)
- Choppiness filter avoids whipsaws by adapting to market regime
- 1d/1w HMA ensures we don't fight the long-term trend
- 6h captures multi-day swings without noise (30-60 trades/year target)
- Different from failed 6h strategies (no weekly pivots, no vol spikes, no KAMA)

Entry conditions (LOOSE to guarantee trades):
- LONG range: CHOP>55 + Fisher<-1.2 + 1w_HMA bullish
- LONG trend: CHOP<45 + Fisher cross above -1.0 + price>1d_HMA>1w_HMA
- SHORT range: CHOP>55 + Fisher>+1.2 + 1w_HMA bearish
- SHORT trend: CHOP<45 + Fisher cross below +1.0 + price<1d_HMA<1w_HMA

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_hma_chop_regime_1d1w_v1"
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

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Formula: Fisher = 0.5 * ln((1 + X) / (1 - X)) where X = normalized price
    Catches reversals when Fisher crosses extreme levels (-1.5, +1.5)
    """
    n = len(high)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    # Calculate typical price
    typical = (high + low + np.roll(high, 1)) / 3.0
    typical[:period] = np.nan
    
    # Normalize price to range -1 to +1
    fisher = np.full(n, np.nan, dtype=np.float64)
    fisher_prev = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        # Find highest high and lowest low over period
        hh = np.nanmax(high[i-period+1:i+1])
        ll = np.nanmin(low[i-period+1:i+1])
        
        price_range = hh - ll
        if price_range > 1e-10:
            # Normalize to 0-1 range
            x = (typical[i] - ll) / price_range
            # Clamp to avoid division by zero
            x = np.clip(x, 0.001, 0.999)
            # Transform to -1 to +1
            x_normalized = 2.0 * x - 1.0
            x_normalized = np.clip(x_normalized, -0.99, 0.99)
            
            # Fisher transform
            if i > period and not np.isnan(fisher[i-1]):
                fisher_prev[i] = fisher[i-1]
                fisher[i] = 0.5 * np.log((1.0 + x_normalized) / (1.0 - x_normalized))
                # Smooth with previous value
                fisher[i] = 0.7 * fisher[i] + 0.3 * fisher_prev[i]
            else:
                fisher[i] = 0.5 * np.log((1.0 + x_normalized) / (1.0 - x_normalized))
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

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    fisher, fisher_prev = calculate_fisher_transform(high, low, period=9)
    
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
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]) or np.isnan(fisher[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop_14[i] > 55.0  # Range market
        is_trending = chop_14[i] < 45.0  # Trend market
        
        # === HTF BIAS (HMA alignment) ===
        hma_1d_bull = close[i] > hma_1d_aligned[i]
        hma_1d_bear = close[i] < hma_1d_aligned[i]
        hma_1w_bull = close[i] > hma_1w_aligned[i]
        hma_1w_bear = close[i] < hma_1w_aligned[i]
        
        # Strong trend alignment
        strong_bull = hma_1d_bull and hma_1w_bull and hma_1d_aligned[i] > hma_1w_aligned[i]
        strong_bear = hma_1d_bear and hma_1w_bear and hma_1d_aligned[i] < hma_1w_aligned[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_cross_up = (not np.isnan(fisher_prev[i]) and 
                          fisher_prev[i] < -1.2 and fisher[i] >= -1.2)
        fisher_cross_down = (not np.isnan(fisher_prev[i]) and 
                            fisher_prev[i] > 1.2 and fisher[i] <= 1.2)
        
        fisher_oversold = fisher[i] < -1.5
        fisher_overbought = fisher[i] > 1.5
        
        # === ENTRY LOGIC (REGIME-ADAPTIVE) ===
        desired_signal = 0.0
        
        if is_choppy:
            # MEAN REVERSION MODE - use Fisher extremes
            # Long when Fisher extremely oversold + HTF bullish bias
            if fisher_oversold and (hma_1w_bull or not hma_1w_bear):
                desired_signal = SIZE_BASE
            # Short when Fisher extremely overbought + HTF bearish bias
            elif fisher_overbought and (hma_1w_bear or not hma_1w_bull):
                desired_signal = -SIZE_BASE
            # Stronger signals on Fisher cross
            elif fisher_cross_up and hma_1w_bull:
                desired_signal = SIZE_STRONG
            elif fisher_cross_down and hma_1w_bear:
                desired_signal = -SIZE_STRONG
        
        elif is_trending:
            # TREND FOLLOWING MODE - use HMA alignment + Fisher confirmation
            # Long in strong uptrend with Fisher confirmation
            if strong_bull and fisher[i] > -1.0 and rsi_14[i] > 45.0:
                desired_signal = SIZE_STRONG
            # Short in strong downtrend with Fisher confirmation
            elif strong_bear and fisher[i] < 1.0 and rsi_14[i] < 55.0:
                desired_signal = -SIZE_STRONG
            # Weaker trend signals
            elif hma_1d_bull and hma_1w_bull and fisher[i] > -0.5:
                desired_signal = SIZE_BASE
            elif hma_1d_bear and hma_1w_bear and fisher[i] < 0.5:
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