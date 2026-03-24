#!/usr/bin/env python3
"""
Experiment #787: 6h Primary + 1d HTF — Fisher Transform Reversal with Trend Filter

Hypothesis: 6h timeframe with Ehlers Fisher Transform captures reversals better than RSI
in bear/range markets (2025 test period). Fisher Transform normalizes price to Gaussian
distribution, making extreme values (-2 to +2) reliable reversal signals. Combined with
1d HMA(21) for trend bias, this should generate 30-60 trades/year with positive Sharpe.

Key innovations:
1. Fisher Transform (period=9) - superior reversal detection vs RSI in choppy markets
2. 1d HMA(21) for HTF trend bias - only trade reversals in trend direction
3. 6h HMA(16/48) for local trend confirmation
4. Fisher extreme thresholds: long when crosses above -1.5, short when crosses below +1.5
5. ATR(14) 2.5x trailing stop for risk management
6. Discrete sizing: 0.0, ±0.25, ±0.30

Entry conditions:
- LONG: 1d HMA bull + Fisher crosses above -1.5 + 6h HMA(16)>HMA(48)
- SHORT: 1d HMA bear + Fisher crosses below +1.5 + 6h HMA(16)<HMA(48)

Target: Sharpe>0.40, trades>=30 train, trades>=3 test, DD>-40%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_hma_trend_1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while maintaining smoothness"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_fisher(close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Excellent for catching reversals in bear/range markets
    Reference: John F. Ehlers, "Rocket Science for Retail Traders"
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.zeros(n)
    fisher[:] = np.nan
    fisher_prev = np.zeros(n)
    fisher_prev[:] = np.nan
    
    # Calculate highest high and lowest low over period
    for i in range(period, n):
        hh = np.max(close[i-period+1:i+1])
        ll = np.min(close[i-period+1:i+1])
        
        # Normalize price to 0-1 range
        if hh > ll:
            x = (close[i] - ll) / (hh - ll)
        else:
            x = 0.5
        
        # Clamp to avoid division by zero
        x = np.clip(x, 0.001, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + x) / (1.0 - x))
        
        if i > period:
            fisher_prev[i] = fisher[i-1]
    
    return fisher, fisher_prev

def calculate_atr(high, low, close, period=14):
    """Average True Range - volatility measure for stops"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 6h indicators
    hma_16 = calculate_hma(close, period=16)
    hma_48 = calculate_hma(close, period=48)
    fisher, fisher_prev = calculate_fisher(close, period=9)
    atr_14 = calculate_atr(high, low, close, period=14)
    
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
        
        if np.isnan(hma_16[i]) or np.isnan(hma_48[i]) or np.isnan(fisher[i]):
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
        
        # === HTF BIAS (1d HMA) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === 6h HMA TREND ===
        hma_6h_bull = hma_16[i] > hma_48[i]
        hma_6h_bear = hma_16[i] < hma_48[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_cross_long = False
        fisher_cross_short = False
        
        if not np.isnan(fisher_prev[i]) and fisher_prev[i] != 0:
            fisher_cross_long = (fisher_prev[i] <= -1.5) and (fisher[i] > -1.5)
            fisher_cross_short = (fisher_prev[i] >= 1.5) and (fisher[i] < 1.5)
        
        # Fisher extreme values for stronger signals
        fisher_extreme_long = fisher[i] < -1.8
        fisher_extreme_short = fisher[i] > 1.8
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: 1d bull + 6h bull + Fisher reversal
        if htf_1d_bull and hma_6h_bull:
            if fisher_cross_long:
                if fisher_extreme_long:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
            elif fisher[i] < -1.5 and hma_16[i] > hma_48[i]:
                # Fisher deeply oversold in uptrend - enter on any pullback
                desired_signal = SIZE_BASE
        
        # SHORT: 1d bear + 6h bear + Fisher reversal
        elif htf_1d_bear and hma_6h_bear:
            if fisher_cross_short:
                if fisher_extreme_short:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
            elif fisher[i] > 1.5 and hma_16[i] < hma_48[i]:
                # Fisher deeply overbought in downtrend - enter on any rally
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