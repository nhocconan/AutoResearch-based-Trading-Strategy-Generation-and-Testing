#!/usr/bin/env python3
"""
Experiment #052: 12h Primary + 1d HTF — Vol Spike Reversion + HMA Trend + BB Mean Revert

Hypothesis: After 51 failed experiments, the clearest pattern from research notes is:
- VOLATILITY SPIKE REVERSION has Sharpe 0.8-1.5 through 2022 crash (BEST EDGE for BTC/ETH)
- ATR(7)/ATR(30) > 1.8 captures panic/extreme volatility spikes
- Price < BB(20, 2.5) lower band during vol spike = panic selling = long opportunity
- Price > BB(20, 2.5) upper band during vol spike = panic buying = short opportunity
- 1d HMA(50) provides major trend bias without being too restrictive
- This works in BOTH 2022 crash (vol spikes down) and 2025 bear (vol spikes up)
- 12h timeframe ensures 20-50 trades/year target (lower fee drag than lower TF)

Key design choices:
- Timeframe: 12h (20-50 trades/year, proven higher TF works better)
- HTF: 1d HMA(50) for major trend bias
- Entry: ATR ratio > 1.8 (vol spike) + BB extreme (2.5 std for wider capture)
- Exit: ATR ratio < 1.3 (vol normalized) OR 2.5x ATR trailing stop
- Position size: 0.30 (30% of capital, conservative for 12h swings)
- LOOSE entry conditions to ensure >=30 trades on train, >=3 on test

Target: Sharpe>0.167 (beat current best), DD>-40%, trades>=30 train, trades>=3 test, ALL symbols Sharpe>0
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_volspike_bb_hma_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
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

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_bollinger_bands(close, period=20, std_mult=2.5):
    """Bollinger Bands with configurable std multiplier"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, lower, sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for major trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (12h) indicators
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, period=20, std_mult=2.5)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Calculate ATR ratio (volatility spike detector)
    atr_ratio = np.zeros(n)
    atr_ratio[:] = np.nan
    for i in range(n):
        if not np.isnan(atr_7[i]) and not np.isnan(atr_30[i]) and atr_30[i] > 1e-10:
            atr_ratio[i] = atr_7[i] / atr_30[i]
    
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size (conservative for 12h)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_ratio[i]) or np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
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
        
        # === VOLATILITY REGIME ===
        # ATR ratio > 1.8 = vol spike (panic/extreme)
        # ATR ratio < 1.3 = vol normalized (calm)
        vol_spike = atr_ratio[i] > 1.8
        vol_normalized = atr_ratio[i] < 1.3
        
        # === HTF BIAS (1d HMA) ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === BOLLINGER BAND EXTREMES ===
        # Price at lower band = oversold (panic selling)
        # Price at upper band = overbought (panic buying)
        bb_width = bb_upper[i] - bb_lower[i]
        if bb_width > 1e-10:
            bb_position = (close[i] - bb_lower[i]) / bb_width
        else:
            bb_position = 0.5
        
        at_bb_lower = bb_position < 0.10  # Very close to lower band
        at_bb_upper = bb_position > 0.90  # Very close to upper band
        
        # === DESIRED SIGNAL (Vol Spike Reversion) ===
        desired_signal = 0.0
        
        # LONG: vol spike + price at BB lower + HTF not strongly bear
        if vol_spike and at_bb_lower and not htf_bear:
            desired_signal = SIZE
        # SHORT: vol spike + price at BB upper + HTF not strongly bull
        elif vol_spike and at_bb_upper and not htf_bull:
            desired_signal = -SIZE
        # FALLBACK LONG: vol spike + price < BB mid + HTF bull (trend pullback)
        elif vol_spike and close[i] < bb_mid[i] and htf_bull:
            desired_signal = SIZE * 0.6
        # FALLBACK SHORT: vol spike + price > BB mid + HTF bear (trend pullback)
        elif vol_spike and close[i] > bb_mid[i] and htf_bear:
            desired_signal = -SIZE * 0.6
        
        # === EXIT SIGNAL (Vol Normalized) ===
        # Close position when volatility normalizes (mean reversion complete)
        if in_position and vol_normalized:
            desired_signal = 0.0
        
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
        elif desired_signal >= SIZE * 0.5:
            final_signal = SIZE * 0.5
        elif desired_signal <= -SIZE * 0.5:
            final_signal = -SIZE * 0.5
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
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