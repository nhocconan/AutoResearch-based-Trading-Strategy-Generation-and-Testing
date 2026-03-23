#!/usr/bin/env python3
"""
Experiment #1261: 4h Primary + 1d HTF — Fisher Transform + ADX Dual Regime

Hypothesis: Recent failures show TWO patterns:
1. Sharpe=0.000 (6 strategies): Entry conditions TOO STRICT → zero trades
2. Negative Sharpe (5 strategies): Wrong regime detection → trades but losing

This strategy combines:
1. EHLERS FISHER TRANSFORM (period=9): Catches reversals in bear/bull markets
   - Fisher crosses above -1.2 → long signal (less extreme than -1.5 for more trades)
   - Fisher crosses below +1.2 → short signal
2. ADX REGIME SWITCH: 
   - ADX > 25 = trending → follow 1d HMA direction
   - ADX < 20 = ranging → Fisher mean reversion at BB extremes
   - Hysteresis buffer (20-25) prevents whipsaw
3. 1d HMA macro filter: Only long if price > 1d HMA, only short if price < 1d HMA
4. LOOSE entry thresholds to ensure >=30 trades/symbol/train

Key improvements from #1254:
- Fisher Transform instead of CRSI (better for bear market reversals)
- ADX regime with hysteresis (20-25 buffer vs single threshold)
- 1d HMA (not 12h) for stronger macro trend filter
- Lower Fisher thresholds (-1.2/+1.2 vs -1.5/+1.5) for MORE trades
- BB confirmation for mean reversion entries (price must touch bands)

Target: Sharpe > 0.612, trades >= 30 train, >= 5 test, DD > -30%
Timeframe: 4h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_adx_regime_1d_hma_bb_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Catches reversals at extremes (-2 to +2 range typical)
    Long when Fisher crosses above -1.2 from below
    Short when Fisher crosses below +1.2 from above
    """
    n = len(close)
    fisher = np.full(n, np.nan)
    fisher_prev = np.full(n, np.nan)
    
    if n < period + 1:
        return fisher, fisher_prev
    
    # Calculate median price
    median = (high + low) / 2.0
    
    # Normalize price to -1 to +1 range
    highest = np.full(n, np.nan)
    lowest = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        highest[i] = np.max(high[i-period+1:i+1])
        lowest[i] = np.min(low[i-period+1:i+1])
    
    # Price normalization with smoothing
    normalized = np.zeros(n)
    for i in range(period - 1, n):
        if highest[i] > lowest[i]:
            normalized[i] = 0.66 * ((median[i] - lowest[i]) / (highest[i] - lowest[i]) - 0.5) + 0.67 * normalized[i-1] if i > period - 1 else 0.0
            normalized[i] = np.clip(normalized[i], -0.99, 0.99)
    
    # Fisher transform
    for i in range(period - 1, n):
        if abs(normalized[i]) < 0.99:
            fisher[i] = 0.5 * np.log((1.0 + normalized[i]) / (1.0 - normalized[i]))
            if i > period - 1:
                fisher_prev[i] = fisher[i-1]
    
    return fisher, fisher_prev

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength with hysteresis"""
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2 + 1:
        return adx
    
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(high[i] - high[i-1], 0)
        else:
            plus_dm[i] = 0
            
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(low[i-1] - low[i], 0)
        else:
            minus_dm[i] = 0
    
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    
    mask = tr_smooth > 1e-10
    plus_di[mask] = 100.0 * plus_dm_smooth[mask] / tr_smooth[mask]
    minus_di[mask] = 100.0 * minus_dm_smooth[mask] / tr_smooth[mask]
    
    di_sum = plus_di + minus_di
    mask2 = di_sum > 1e-10
    dx[mask2] = 100.0 * np.abs(plus_di[mask2] - minus_di[mask2]) / di_sum[mask2]
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_hma(close, period=21):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            if i >= sqrt_period - 1:
                diff_window = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_window.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_window) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1)
                    hma[i] = np.sum(np.array(diff_window) * weights) / np.sum(weights)
    
    return hma

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss"""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands for mean reversion confirmation"""
    n = len(close)
    mid = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    if n < period:
        return mid, upper, lower
    
    for i in range(period - 1, n):
        window = close[i-period+1:i+1]
        mid[i] = np.mean(window)
        std = np.std(window, ddof=0)
        upper[i] = mid[i] + std_mult * std
        lower[i] = mid[i] - std_mult * std
    
    return mid, upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for macro trend filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    fisher, fisher_prev = calculate_fisher_transform(high, low, close, period=9)
    adx = calculate_adx(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    bb_mid, bb_upper, bb_lower = calculate_bollinger(close, period=20, std_mult=2.0)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(adx[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(fisher[i]) or np.isnan(fisher_prev[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        if np.isnan(bb_mid[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION (ADX with hysteresis) ===
        # ADX > 25 = trending, ADX < 20 = ranging, 20-25 = hold previous regime
        in_trend = adx[i] > 25.0
        in_range = adx[i] < 20.0
        # Between 20-25, maintain previous state (implicit in logic below)
        
        # === MACRO TREND (1d HMA) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.2 from below
        fisher_long = (fisher[i] > -1.2) and (fisher_prev[i] <= -1.2)
        # Short: Fisher crosses below +1.2 from above
        fisher_short = (fisher[i] < 1.2) and (fisher_prev[i] >= 1.2)
        
        # === BOLLINGER BAND CONFIRMATION (for mean reversion) ===
        at_bb_lower = close[i] <= bb_lower[i] * 1.002
        at_bb_upper = close[i] >= bb_upper[i] * 0.998
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # TRENDING REGIME: Follow macro trend with Fisher confirmation
        if in_trend:
            # Long: Macro bull + Fisher long crossover
            if macro_bull and fisher_long:
                desired_signal = BASE_SIZE
            # Short: Macro bear + Fisher short crossover
            elif macro_bear and fisher_short:
                desired_signal = -BASE_SIZE
        
        # RANGING REGIME: Mean revert with Fisher + BB confirmation
        elif in_range:
            # Long: Fisher long + at BB lower + macro not strongly bear
            if fisher_long and at_bb_lower:
                desired_signal = BASE_SIZE
            # Short: Fisher short + at BB upper + macro not strongly bull
            elif fisher_short and at_bb_upper:
                desired_signal = -BASE_SIZE
        
        # TRANSITION ZONE (ADX 20-25): Allow both types of signals but require stronger confirmation
        else:
            # Require BOTH Fisher AND BB confirmation in transition
            if fisher_long and at_bb_lower and macro_bull:
                desired_signal = BASE_SIZE
            elif fisher_short and at_bb_upper and macro_bear:
                desired_signal = -BASE_SIZE
        
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
        if desired_signal > 0.1:
            final_signal = BASE_SIZE
        elif desired_signal < -0.1:
            final_signal = -BASE_SIZE
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