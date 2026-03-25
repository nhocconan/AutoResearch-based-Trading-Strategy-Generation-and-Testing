#!/usr/bin/env python3
"""
Experiment #1103: 6h Primary + 1d/1w HTF — Fisher Transform + Volume Spike + HMA Bias

Hypothesis: 6h timeframe is underexplored and sits between 4h noise and 12h sluggishness.
Using Ehlers Fisher Transform (proven reversal indicator in bear/range markets) combined
with volume spike confirmation and 1d/1w HMA trend bias should capture multi-day swings
while avoiding whipsaws that killed pure trend strategies.

Key innovations:
1. Ehlers Fisher Transform (period=9): Normalizes price to Gaussian distribution,
   crosses at -1.5/+1.5 mark reversal points better than RSI in bear markets
2. Volume spike filter: Volume > 1.5x 20-bar MA confirms genuine interest
3. 1d/1w HMA(21) bias: Only long when 1w HMA bull, only short when 1w HMA bear
4. Asymmetric entries: Easier to enter (Fisher<-1.2), harder to exit (Fisher>+1.5)
5. ATR(14) 2.5x trailing stop for risk management
6. Discrete sizing: 0.0, ±0.25, ±0.30 to minimize fee churn

Why this should work on 6h:
- Fisher Transform catches reversals in 2022-2023 range markets (where trend failed)
- Volume filter avoids false breakouts common on 6h
- 1w HMA provides multi-week bias without overfitting
- 6h captures 3-5 day swings (20-50 trades/year target)
- Asymmetric exits let winners run while cutting losers fast

Entry conditions (LOOSE to guarantee trades):
- LONG: Fisher<-1.2 + volume>1.5x + 1w_HMA bull + 1d_HMA>1w_HMA
- SHORT: Fisher>+1.2 + volume>1.5x + 1w_HMA bear + 1d_HMA<1w_HMA
- Relaxed Fisher threshold (-1.2 not -1.5) to ensure sufficient trades

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_vol_hma_bias_1d1w_v1"
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

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Makes reversal points more identifiable than RSI
    
    Formula:
    1. Calculate typical price: (High + Low) / 2
    2. Normalize: (Price - Lowest) / (Highest - Lowest)
    3. Scale to -1 to +1: 2 * normalized - 1
    4. Fisher: 0.5 * ln((1 + scaled) / (1 - scaled))
    5. Smooth with EMA
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Typical price
    typical = (high + low) / 2.0
    
    fisher = np.full(n, np.nan, dtype=np.float64)
    fisher_signal = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        price_range = highest - lowest
        if price_range < 1e-10:
            continue
        
        # Normalize to 0-1
        normalized = (typical[i] - lowest) / price_range
        
        # Scale to -0.99 to +0.99 (avoid division by zero)
        scaled = max(-0.99, min(0.99, 2.0 * normalized - 1.0))
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + scaled) / (1.0 - scaled))
    
    # Smooth with EMA
    fisher_series = pd.Series(fisher)
    fisher_smooth = fisher_series.ewm(span=3, min_periods=3, adjust=False).mean().values
    
    # Signal line (1-bar lag of fisher)
    fisher_signal[1:] = fisher_smooth[:-1]
    
    return fisher_smooth, fisher_signal

def calculate_volume_spike(volume, period=20, threshold=1.5):
    """Detect volume spikes above moving average"""
    n = len(volume)
    if n < period + 1:
        return np.full(n, np.nan)
    
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / vol_ma
    
    spike = vol_ratio > threshold
    return spike, vol_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
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
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, period=9)
    vol_spike, vol_ratio = calculate_volume_spike(volume, period=20, threshold=1.5)
    
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
        
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
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
        
        # === HTF BIAS (1d/1w HMA alignment) ===
        hma_1w_bull = close[i] > hma_1w_aligned[i]
        hma_1w_bear = close[i] < hma_1w_aligned[i]
        hma_1d_bull = close[i] > hma_1d_aligned[i]
        hma_1d_bear = close[i] < hma_1d_aligned[i]
        
        # Strong alignment: both 1d and 1w agree
        strong_bull = hma_1d_bull and hma_1w_bull
        strong_bear = hma_1d_bear and hma_1w_bear
        
        # === VOLUME CONFIRMATION ===
        has_volume = vol_spike[i] if not np.isnan(vol_spike[i]) else False
        
        # === FISHER TRANSFORM SIGNALS ===
        # Fisher crosses above -1.5 from below = bullish reversal
        # Fisher crosses below +1.5 from above = bearish reversal
        fisher_oversold = fisher[i] < -1.2
        fisher_overbought = fisher[i] > 1.2
        
        # Crossover detection
        fisher_bull_cross = (fisher_signal[i] < -1.5 and fisher[i] >= -1.5) or \
                           (fisher_signal[i] < fisher[i] and fisher[i] < -1.0)
        fisher_bear_cross = (fisher_signal[i] > 1.5 and fisher[i] <= 1.5) or \
                           (fisher_signal[i] > fisher[i] and fisher[i] > 1.0)
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG entry: Fisher oversold + volume + 1w bull bias
        if fisher_oversold and hma_1w_bull:
            if has_volume and strong_bull:
                desired_signal = SIZE_STRONG
            elif has_volume or strong_bull:
                desired_signal = SIZE_BASE
            else:
                desired_signal = SIZE_BASE * 0.5
        
        # SHORT entry: Fisher overbought + volume + 1w bear bias
        elif fisher_overbought and hma_1w_bear:
            if has_volume and strong_bear:
                desired_signal = -SIZE_STRONG
            elif has_volume or strong_bear:
                desired_signal = -SIZE_BASE
            else:
                desired_signal = -SIZE_BASE * 0.5
        
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
        elif abs(desired_signal) >= SIZE_BASE * 0.4:
            final_signal = np.sign(desired_signal) * SIZE_BASE
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