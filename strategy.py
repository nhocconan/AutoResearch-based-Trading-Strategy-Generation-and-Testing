#!/usr/bin/env python3
"""
Experiment #040: 1h Primary + 4h/12h HTF — Fisher Transform + HMA Trend + Volume

Hypothesis: After 39 experiments, 1h strategies fail due to: (1) too many trades causing fee drag, 
(2) too strict filters causing 0 trades, (3) RSI doesn't catch reversals well in bear/range markets.

This strategy uses:
1. 12h HMA for primary trend bias (slow, stable direction filter)
2. 4h HMA for secondary confirmation (medium-term momentum)
3. Fisher Transform for entry timing (superior to RSI for reversals - Gaussian normalized)
4. Volume spike confirmation (>1.1x 20-period avg) - loose filter
5. Position size: 0.25 (smaller for 1h to reduce fee impact)
6. 2.5x ATR trailing stop for risk management

Key insight: Fisher Transform normalizes price to Gaussian distribution, making extreme 
values (-1.5 to +1.5) reliable reversal signals. Combined with dual HTF trend filter, 
this should generate 40-80 trades/year with good win rate in bear/range markets.

Entry Logic (LOOSE for trade generation):
- Long: 12h_HMA_bull + (4h_HMA_bull OR Fisher<-1.2) + Volume>1.1x
- Short: 12h_HMA_bear + (4h_HMA_bear OR Fisher>+1.2) + Volume>1.1x

Target: Sharpe>0.3, trades>30/symbol train, >3/symbol test, DD>-40%
Timeframe: 1h (target 40-80 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_hma_vol_4h12h_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - smoother and more responsive than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(data, span):
        result = np.full(len(data), np.nan)
        weights = np.arange(1, span + 1, dtype=np.float64)
        for i in range(span - 1, len(data)):
            window = data[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half_period)
    wma_full = wma(close, period)
    double_wma_half = 2.0 * wma_half - wma_full
    hma = wma(double_wma_half, sqrt_period)
    
    return hma

def calculate_fisher(close, high, low, period=10):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution.
    Extreme values (-2 to +2) indicate reversal points.
    Better than RSI for catching turns in bear/range markets.
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    fisher = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        # Find highest high and lowest low over lookback
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        
        # Normalize price to -1 to +1 range
        range_val = highest - lowest
        if range_val < 1e-10:
            continue
        
        x = (2.0 * (close[i] - lowest) / range_val) - 1.0
        
        # Clamp to avoid division by zero in log
        x = np.clip(x, -0.999, 0.999)
        
        # Fisher Transform formula
        fisher[i] = 0.5 * np.log((1.0 + x) / (1.0 - x))
    
    return fisher

def calculate_atr(high, low, close, period=14):
    """Average True Range - for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_ma(volume, period=20):
    """Volume moving average for confirmation"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_ma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align HTF HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (1h) indicators
    hma_1h = calculate_hma(close, period=21)
    hma_1h_fast = calculate_hma(close, period=9)
    fisher = calculate_fisher(close, high, low, period=10)
    atr = calculate_atr(high, low, close, period=14)
    vol_ma = calculate_volume_ma(volume, period=20)
    
    signals = np.zeros(n)
    SIZE = 0.25  # Smaller size for 1h to reduce fee impact
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1h[i]) or np.isnan(fisher[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (12h HMA) - PRIMARY TREND ===
        hma_12h_bull = close[i] > hma_12h_aligned[i]
        hma_12h_bear = close[i] < hma_12h_aligned[i]
        
        # === 4h TREND CONFIRMATION - SECONDARY ===
        hma_4h_bull = close[i] > hma_4h_aligned[i]
        hma_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === 1h MOMENTUM (fast HMA vs slow HMA) ===
        hma_1h_bull = hma_1h_fast[i] > hma_1h[i] if not np.isnan(hma_1h_fast[i]) else False
        hma_1h_bear = hma_1h_fast[i] < hma_1h[i] if not np.isnan(hma_1h_fast[i]) else False
        
        # === FISHER EXTREMES (reversal signals) ===
        fisher_oversold = fisher[i] < -1.2  # Loose threshold for more trades
        fisher_overbought = fisher[i] > 1.2
        
        # === VOLUME CONFIRMATION (loose filter) ===
        volume_confirmed = volume[i] > 1.1 * vol_ma[i] if vol_ma[i] > 0 else True
        
        # === DESIRED SIGNAL (LOOSE for trade generation) ===
        desired_signal = 0.0
        
        # LONG: 12h bull + (4h bull OR Fisher oversold) + Volume
        # This ensures we trade with HTF trend but can enter on pullbacks
        if hma_12h_bull:
            if hma_4h_bull and volume_confirmed:
                # Strong uptrend - enter
                desired_signal = SIZE
            elif fisher_oversold and volume_confirmed:
                # Pullback in uptrend - buy the dip using Fisher
                desired_signal = SIZE
        
        # SHORT: 12h bear + (4h bear OR Fisher overbought) + Volume
        # This ensures we trade with HTF trend but can enter on rallies
        if hma_12h_bear:
            if hma_4h_bear and volume_confirmed:
                # Strong downtrend - enter
                desired_signal = -SIZE
            elif fisher_overbought and volume_confirmed:
                # Rally in downtrend - sell the rip using Fisher
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals