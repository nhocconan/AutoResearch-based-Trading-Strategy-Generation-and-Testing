#!/usr/bin/env python3
"""
Experiment #1060: 6h Primary + 1d/1w HTF — Fisher Transform + Choppiness Regime + HMA Bias

Hypothesis: 6h timeframe is underexplored middle ground between 4h and 12h. Using Ehlers Fisher
Transform for reversal detection combined with Choppiness Index regime filter and 1d/1w HMA bias
will capture both mean-reversion in choppy markets and trend continuations in trending markets.

Key innovations:
1. Ehlers Fisher Transform (period=9): Normalizes price to Gaussian distribution, extreme values
   (-1.5, +1.5) mark high-probability reversals. Works well in bear/range markets (2022-2023).
2. Choppiness Index (CHOP 14): >61.8 = range (favor mean reversion), <38.2 = trend (favor continuation)
3. 1d HMA(21) + 1w HMA(21): Long-term bias filter - only long when price > 1w_HMA, vice versa
4. Regime-adaptive entries:
   - Choppy (CHOP>55): Fisher extremes trigger mean reversion (Fisher<-1.5 long, >+1.5 short)
   - Trending (CHOP<45): Fisher pullback to zero + HTF alignment triggers trend continuation
5. RSI(14) confirmation filter: Prevents entries against strong momentum
6. ATR(14) 2.5x trailing stop for risk management
7. Discrete sizing: 0.0, ±0.25, ±0.30 to minimize fee churn

Why 6h should work:
- 6h captures multi-day swings (4 bars/day) without 4h noise or 12h slowness
- Fisher Transform excels at catching reversals in bear markets (2022 crash, 2023 range)
- Choppiness filter avoids trend-following whipsaws in sideways markets
- 1d/1w HMA ensures we trade with long-term trend direction
- LOOSE entry conditions guarantee trades (Fisher<-1.2 OR <-1.5, RSI<45 not <30)

Entry conditions (LOOSE to guarantee >=30 trades/year):
- LONG choppy: CHOP>50 + Fisher<-1.2 + price>1w_HMA*0.90 + RSI<50
- LONG trending: CHOP<50 + Fisher<-0.5 + price>1d_HMA>1w_HMA + RSI>40
- SHORT choppy: CHOP>50 + Fisher>+1.2 + price<1w_HMA*1.10 + RSI>50
- SHORT trending: CHOP<50 + Fisher>+0.5 + price<1d_HMA<1w_HMA + RSI<60

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_chop_hma_regime_1d1w_v2"
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

def calculate_fisher(close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Formula: Fisher = 0.5 * ln((1 + X) / (1 - X)) where X = 0.66 * ((price - LL) / (HH - LL) - 0.5)
    Extreme values (-1.5, +1.5) mark high-probability reversals
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    fisher = np.full(n, np.nan, dtype=np.float64)
    fisher_prev = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        highest = np.max(close[i-period+1:i+1])
        lowest = np.min(close[i-period+1:i+1])
        
        price_range = highest - lowest
        if price_range < 1e-10:
            continue
        
        x = 0.66 * ((close[i] - lowest) / price_range - 0.5)
        x = np.clip(x, -0.999, 0.999)  # Prevent division by zero
        
        fisher[i] = 0.5 * np.log((1 + x) / (1 - x))
        
        if i > period and not np.isnan(fisher[i-1]):
            fisher_prev[i] = fisher[i-1]
        else:
            fisher_prev[i] = fisher[i]
    
    return fisher, fisher_prev

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    Formula: 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
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
    fisher, fisher_prev = calculate_fisher(close, period=9)
    
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
        is_choppy = chop_14[i] > 50.0  # Range market (slightly relaxed from 55)
        is_trending = chop_14[i] < 50.0  # Trend market
        
        # === HTF BIAS (HMA alignment) ===
        hma_1d_bull = close[i] > hma_1d_aligned[i]
        hma_1d_bear = close[i] < hma_1d_aligned[i]
        hma_1w_bull = close[i] > hma_1w_aligned[i] * 0.95  # 5% buffer
        hma_1w_bear = close[i] < hma_1w_aligned[i] * 1.05  # 5% buffer
        
        # Strong trend alignment
        strong_bull = hma_1d_bull and hma_1w_bull and hma_1d_aligned[i] > hma_1w_aligned[i]
        strong_bear = hma_1d_bear and hma_1w_bear and hma_1d_aligned[i] < hma_1w_aligned[i]
        
        # === ENTRY LOGIC (REGIME-ADAPTIVE) ===
        desired_signal = 0.0
        
        if is_choppy:
            # MEAN REVERSION MODE - use Fisher Transform extremes
            # Long when Fisher extremely oversold + weekly bias OK
            if fisher[i] < -1.2 and hma_1w_bull and rsi_14[i] < 55:
                desired_signal = SIZE_BASE
            # Short when Fisher extremely overbought + weekly bias OK
            elif fisher[i] > 1.2 and hma_1w_bear and rsi_14[i] > 45:
                desired_signal = -SIZE_BASE
            # Stronger signals at more extreme Fisher
            elif fisher[i] < -1.5 and hma_1w_bull:
                desired_signal = SIZE_STRONG
            elif fisher[i] > 1.5 and hma_1w_bear:
                desired_signal = -SIZE_STRONG
        
        elif is_trending:
            # TREND FOLLOWING MODE - use Fisher pullback + HMA alignment
            # Long in uptrend when Fisher pulls back from oversold
            if strong_bull and fisher[i] < 0.5 and fisher[i] > -1.5 and rsi_14[i] > 40:
                desired_signal = SIZE_STRONG
            # Short in downtrend when Fisher pulls back from overbought
            elif strong_bear and fisher[i] > -0.5 and fisher[i] < 1.5 and rsi_14[i] < 60:
                desired_signal = -SIZE_STRONG
            # Weaker trend signals
            elif hma_1d_bull and hma_1w_bull and fisher[i] < 0.0 and rsi_14[i] > 45:
                desired_signal = SIZE_BASE
            elif hma_1d_bear and hma_1w_bear and fisher[i] > 0.0 and rsi_14[i] < 55:
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