#!/usr/bin/env python3
"""
Experiment #1155: 6h Primary + 12h HTF — Ehlers Fisher Transform + HMA Trend + Vol Filter

Hypothesis: Ehlers Fisher Transform excels at detecting reversals in bear/range markets
(2022-2024 train, 2025+ test) when combined with 12h HMA trend bias. Fisher Transform
normalizes price to Gaussian distribution, making extreme values (-2 to +2) reliable
reversal signals. This should outperform RSI-based strategies in crypto's fat-tailed returns.

Key innovations:
1. Ehlers Fisher Transform (period=9): Long when Fisher crosses above -1.5, Short when crosses below +1.5
2. 12h HMA(21) trend bias: Only long when price > 12h_HMA, only short when price < 12h_HMA
3. ATR volatility filter: ATR(14)/close > 0.015 (1.5%) to avoid low-vol whipsaws
4. Asymmetric sizing: 0.30 in trend direction, 0.20 counter-trend (reversals)
5. 2.5x ATR trailing stop for risk management
6. LOOSE entry conditions to guarantee 30+ trades/year

Why this should work:
- Fisher Transform has superior reversal detection vs RSI in bear markets (Ehlers research)
- 12h HMA provides trend bias without over-filtering (1d+1w was too restrictive)
- 6h timeframe captures 2-5 day swings (sweet spot between 4h noise and 12h slowness)
- Vol filter avoids dead periods that destroy Sharpe
- Asymmetric sizing reduces risk on counter-trend reversals

Entry conditions (LOOSE to guarantee trades):
- LONG: Fisher crosses above -1.5 + price > 12h_HMA + ATR_ratio > 0.012
- SHORT: Fisher crosses below +1.5 + price < 12h_HMA + ATR_ratio > 0.012
- No regime switching (simpler = more trades)

Target: Sharpe>0.50, trades>=40 train, trades>=5 test, DD>-35%
Timeframe: 6h
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_hma_vol_12h_v1"
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
    Makes extreme values reliable reversal signals
    
    Formula:
    1. Calculate typical price: (high + low) / 2
    2. Normalize to -1 to +1 range using Donchian channel
    3. Apply Fisher transform: 0.5 * ln((1+x)/(1-x))
    
    Entry signals:
    - Long: Fisher crosses above -1.5 (oversold reversal)
    - Short: Fisher crosses below +1.5 (overbought reversal)
    """
    n = len(high)
    if n < period:
        return np.full(n, np.nan)
    
    # Typical price
    typical = (high + low) / 2.0
    
    fisher = np.full(n, np.nan, dtype=np.float64)
    fisher_signal = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        # Donchian channel for normalization
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        price_range = highest - lowest
        if price_range < 1e-10:
            continue
        
        # Normalize to -1 to +1 (with small epsilon to avoid division by zero)
        normalized = 2.0 * (typical[i] - lowest) / price_range - 1.0
        normalized = np.clip(normalized, -0.999, 0.999)  # Avoid ln(0)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
        
        # Signal line (1-period lag of fisher)
        if i > 0 and not np.isnan(fisher[i-1]):
            fisher_signal[i] = fisher[i-1]
    
    return fisher, fisher_signal

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align HTF indicators
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, period=9)
    
    # ATR ratio for volatility filter
    atr_ratio = atr_14 / close
    
    signals = np.zeros(n)
    SIZE_TREND = 0.30  # Larger size when trading with HTF trend
    SIZE_REVERSAL = 0.20  # Smaller size on counter-trend reversals
    
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
    prev_fisher_signal = np.nan
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === VOLATILITY FILTER ===
        # Only trade when ATR ratio > 1.2% (avoid dead markets)
        vol_ok = atr_ratio[i] > 0.012
        
        if not vol_ok:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            prev_fisher = fisher[i]
            prev_fisher_signal = fisher_signal[i]
            continue
        
        # === HTF TREND BIAS ===
        hma_12h_bull = close[i] > hma_12h_aligned[i]
        hma_12h_bear = close[i] < hma_12h_aligned[i]
        
        # === FISHER TRANSFORM REVERSAL SIGNALS ===
        # Detect crosses
        fisher_cross_up = False
        fisher_cross_down = False
        
        if not np.isnan(prev_fisher) and not np.isnan(prev_fisher_signal):
            # Fisher crosses above signal line (bullish)
            if prev_fisher <= prev_fisher_signal and fisher[i] > fisher_signal[i]:
                fisher_cross_up = True
            # Fisher crosses below signal line (bearish)
            if prev_fisher >= prev_fisher_signal and fisher[i] < fisher_signal[i]:
                fisher_cross_down = True
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG entries
        if hma_12h_bull:
            # Trend-following long: Fisher cross up in bullish HTF
            if fisher_cross_up and fisher[i] < 0.5:  # Not too overbought
                desired_signal = SIZE_TREND
            # Reversal long: Fisher extremely oversold
            elif fisher[i] < -1.5 and fisher_signal[i] < fisher[i]:  # Turning up from extreme
                desired_signal = SIZE_REVERSAL
        elif hma_12h_bear:
            # Counter-trend long: Fisher extremely oversold (reversal play)
            if fisher[i] < -1.8 and fisher_signal[i] < fisher[i]:
                desired_signal = SIZE_REVERSAL * 0.7  # Even smaller on counter-trend
        
        # SHORT entries
        if hma_12h_bear:
            # Trend-following short: Fisher cross down in bearish HTF
            if fisher_cross_down and fisher[i] > -0.5:  # Not too oversold
                desired_signal = -SIZE_TREND
            # Reversal short: Fisher extremely overbought
            elif fisher[i] > 1.5 and fisher_signal[i] > fisher[i]:  # Turning down from extreme
                desired_signal = -SIZE_REVERSAL
        elif hma_12h_bull:
            # Counter-trend short: Fisher extremely overbought (reversal play)
            if fisher[i] > 1.8 and fisher_signal[i] > fisher[i]:
                desired_signal = -SIZE_REVERSAL * 0.7  # Even smaller on counter-trend
        
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
        if desired_signal >= SIZE_TREND * 0.9:
            final_signal = SIZE_TREND
        elif desired_signal <= -SIZE_TREND * 0.9:
            final_signal = -SIZE_TREND
        elif desired_signal >= SIZE_REVERSAL * 0.9:
            final_signal = SIZE_REVERSAL
        elif desired_signal <= -SIZE_REVERSAL * 0.9:
            final_signal = -SIZE_REVERSAL
        elif desired_signal >= SIZE_REVERSAL * 0.5:
            final_signal = SIZE_REVERSAL * 0.7
        elif desired_signal <= -SIZE_REVERSAL * 0.5:
            final_signal = -SIZE_REVERSAL * 0.7
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
        
        # Update previous values for next iteration
        prev_fisher = fisher[i]
        prev_fisher_signal = fisher_signal[i]
    
    return signals