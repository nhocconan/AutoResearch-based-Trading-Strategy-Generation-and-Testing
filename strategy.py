#!/usr/bin/env python3
"""
Experiment #1140: 6h Primary + 1d/1w HTF — Fisher Transform Reversals + Volume Confirmation

Hypothesis: The Ehlers Fisher Transform excels at catching reversals in bear/range markets
(2022-2024) where RSI fails. Combined with 1w HMA trend filter and volume spike confirmation,
this should generate profitable mean-reversion trades on 6h timeframe.

Key innovations:
1. Ehlers Fisher Transform (period=9): Normalizes price to Gaussian distribution, 
   crosses at ±1.5 signal reversals better than RSI extremes
2. Volume spike filter: volume > 1.5 * SMA(volume, 20) confirms real moves vs noise
3. 1w HMA(21) as primary trend bias (more stable than 1d for 6h entries)
4. 1d HMA(21) as secondary confirmation filter
5. Asymmetric entries: Long only when price > 1w_HMA*0.92, Short only when price < 1w_HMA*1.08
6. ATR(14) 2.5x trailing stop for risk management

Why 6h should work:
- Captures 2-4 day swings (perfect for crypto mean reversion)
- Less noise than 4h, more trades than 12h
- 30-60 trades/year target with strict filters
- Fisher Transform proven in literature for reversal detection

Entry conditions (LOOSE to guarantee trades):
- LONG: Fisher < -1.2 + Fisher crossing up + volume_spike + price > 1w_HMA*0.90
- SHORT: Fisher > +1.2 + Fisher crossing down + volume_spike + price < 1w_HMA*1.10

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_vol_1d1w_v1"
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
    Excellent for detecting reversals in ranging/bear markets
    
    Formula:
    1. Calculate typical price: (High + Low) / 2
    2. Normalize: 0.66 * ((TP - LL) / (HH - LL) - 0.5) + 0.66 * prev_norm
    3. Fisher: 0.5 * ln((1 + norm) / (1 - norm)) + 0.5 * prev_fisher
    
    Crosses above -1.5 = long signal
    Crosses below +1.5 = short signal
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    typical_price = (high + low) / 2.0
    
    # Normalize price within lookback window
    normalized = np.full(n, np.nan, dtype=np.float64)
    fisher = np.full(n, np.nan, dtype=np.float64)
    trigger = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        # Find highest high and lowest low over lookback period
        hh = np.nanmax(high[i-period+1:i+1])
        ll = np.nanmin(low[i-period+1:i+1])
        
        price_range = hh - ll
        if price_range < 1e-10:
            normalized[i] = 0.0
        else:
            # Ehlers normalization formula
            normalized[i] = 0.66 * ((typical_price[i] - ll) / price_range - 0.5) + \
                           0.67 * normalized[i-1] if not np.isnan(normalized[i-1]) else 0.0
        
        # Clamp normalized value to prevent division by zero
        normalized[i] = np.clip(normalized[i], -0.999, 0.999)
        
        # Fisher Transform calculation
        if not np.isnan(normalized[i]):
            fisher[i] = 0.5 * np.log((1.0 + normalized[i]) / (1.0 - normalized[i])) + \
                       0.5 * fisher[i-1] if not np.isnan(fisher[i-1]) else 0.0
            
            # Trigger line (previous Fisher value)
            trigger[i] = fisher[i-1] if not np.isnan(fisher[i-1]) else fisher[i]
    
    return fisher, trigger

def calculate_volume_spike(volume, period=20, threshold=1.5):
    """Detect volume spikes above moving average"""
    n = len(volume)
    if n < period:
        return np.full(n, False)
    
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    spike = volume > (threshold * vol_sma)
    spike[:period] = False
    return spike

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
    fisher, trigger = calculate_fisher_transform(high, low, close, period=9)
    vol_spike = calculate_volume_spike(volume, period=20, threshold=1.5)
    
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
        
        if np.isnan(fisher[i]) or np.isnan(trigger[i]):
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
        
        # === HTF TREND BIAS ===
        hma_1w_bull = close[i] > hma_1w_aligned[i] * 0.92
        hma_1w_bear = close[i] < hma_1w_aligned[i] * 1.08
        hma_1d_bull = close[i] > hma_1d_aligned[i]
        hma_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_cross_up = fisher[i] > -1.2 and trigger[i] <= -1.2
        fisher_cross_down = fisher[i] < 1.2 and trigger[i] >= 1.2
        fisher_extreme_low = fisher[i] < -1.5
        fisher_extreme_high = fisher[i] > 1.5
        
        # === ENTRY LOGIC (MEAN REVERSION WITH TREND FILTER) ===
        desired_signal = 0.0
        
        # LONG entries - Fisher oversold + volume confirmation + HTF support
        if fisher_cross_up or fisher_extreme_low:
            if hma_1w_bull and vol_spike[i]:
                desired_signal = SIZE_BASE
            elif hma_1w_bull and hma_1d_bull:
                desired_signal = SIZE_STRONG
        
        # SHORT entries - Fisher overbought + volume confirmation + HTF resistance
        if fisher_cross_down or fisher_extreme_high:
            if hma_1w_bear and vol_spike[i]:
                desired_signal = -SIZE_BASE
            elif hma_1w_bear and hma_1d_bear:
                desired_signal = -SIZE_STRONG
        
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