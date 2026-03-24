#!/usr/bin/env python3
"""
Experiment #995: 6h Primary + 1d/1w HTF — Fisher Transform + Vol Spike Reversion

Hypothesis: 6h timeframe with Ehlers Fisher Transform for reversal detection + 
ATR vol spike filter + 1d/1w trend bias will outperform in mixed 2022-2025 markets.

Key innovations:
1. Ehlers Fisher Transform (period=9): Normalizes price to Gaussian distribution
   - Fisher < -1.5 = oversold reversal signal (long)
   - Fisher > +1.5 = overbought reversal signal (short)
2. Vol Spike Filter: ATR(7)/ATR(30) > 1.3 captures panic/reversal points
3. 1d HMA(21) for intermediate trend bias
4. 1w momentum (close > open) for weekly bias
5. Asymmetric entries: Only long in bull HTF, only short in bear HTF
6. ATR(14) 2.5x trailing stop for risk management

Why this should work:
- Fisher Transform catches reversals better than RSI in bear markets (2022, 2025)
- Vol spike filter enters after panic, avoids catching falling knives
- HTF bias prevents counter-trend trades in strong moves
- 6h captures multi-day swings without 4h noise or 12h lag
- DIFFERENT from failed CHOP+CRSI approach - uses Gaussian normalization

Entry conditions (LOOSE to guarantee trades):
- LONG = 1w bull + 1d bull + Fisher < -1.2 + ATR_ratio > 1.3
- SHORT = 1w bear + 1d bear + Fisher > +1.2 + ATR_ratio > 1.3
- Also: Fisher cross above -1.0 (long) or below +1.0 (short) for continuation

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_volspike_regime_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform
    Normalizes price to Gaussian distribution for clearer reversal signals
    
    Steps:
    1. Calculate typical price: (high + low) / 2
    2. Normalize to -1 to +1 range using highest high / lowest low over period
    3. Apply Fisher transform: 0.5 * ln((1+x)/(1-x))
    4. Smooth with 1-period lag for signal line
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.full(n, np.nan, dtype=np.float64)
    fisher_signal = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        # Typical price
        typical = (high[i] + low[i]) / 2.0
        
        # Find highest high and lowest low over period
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        # Normalize to -1 to +1 (with bounds check)
        if highest_high > lowest_low:
            normalized = 2.0 * (typical - lowest_low) / (highest_high - lowest_low) - 1.0
            normalized = max(-0.999, min(0.999, normalized))  # Prevent division by zero
            
            # Fisher transform
            fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
            
            # Signal line (1-period lag)
            if i > period:
                fisher_signal[i] = fisher[i-1]
            else:
                fisher_signal[i] = fisher[i]
    
    return fisher, fisher_signal

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

def calculate_hma(close, period):
    """Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=np.float64)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1].astype(np.float64)
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

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
    
    # Weekly momentum: close vs open
    weekly_momentum_raw = (df_1w['close'].values - df_1w['open'].values) / (df_1w['open'].values + 1e-10)
    weekly_momentum_aligned = align_htf_to_ltf(prices, df_1w, weekly_momentum_raw)
    
    # Calculate 6h indicators
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, period=9)
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    rsi_14 = calculate_rsi(close, period=14)
    
    # ATR ratio for vol spike detection
    atr_ratio = np.full(n, np.nan, dtype=np.float64)
    for i in range(30, n):
        if atr_30[i] > 1e-10 and not np.isnan(atr_7[i]):
            atr_ratio[i] = atr_7[i] / atr_30[i]
    
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
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(weekly_momentum_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(atr_ratio[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1w momentum + 1d HMA) ===
        htf_1w_bull = weekly_momentum_aligned[i] > 0.0
        htf_1w_bear = weekly_momentum_aligned[i] < 0.0
        
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === VOL SPIKE FILTER ===
        vol_spike = atr_ratio[i] > 1.3  # Current vol > 1.3x average
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_oversold = fisher[i] < -1.2  # Reversal long signal
        fisher_overbought = fisher[i] > 1.2  # Reversal short signal
        
        # Fisher cross signals (for continuation entries)
        fisher_cross_long = False
        fisher_cross_short = False
        if i > 0 and not np.isnan(fisher[i-1]) and not np.isnan(fisher_signal[i-1]):
            # Cross above -1.0 (bullish continuation)
            fisher_cross_long = (fisher[i-1] < -1.0) and (fisher[i] >= -1.0)
            # Cross below +1.0 (bearish continuation)
            fisher_cross_short = (fisher[i-1] > 1.0) and (fisher[i] <= 1.0)
        
        # === RSI FILTER (avoid extreme counter-trend) ===
        rsi_neutral = 30 < rsi_14[i] < 70
        
        # === ENTRY LOGIC (ASYMMETRIC - HTF DIRECTION ONLY) ===
        desired_signal = 0.0
        
        # LONG entries (only when HTF is bullish)
        if htf_1w_bull and htf_1d_bull:
            if fisher_oversold and vol_spike:
                # Primary: Fisher oversold + vol spike = panic reversal
                desired_signal = SIZE_STRONG
            elif fisher_cross_long and rsi_14[i] < 50:
                # Continuation: Fisher cross + RSI not overbought
                desired_signal = SIZE_BASE
            elif fisher[i] < -0.5 and rsi_14[i] < 40:
                # Deep pullback in uptrend
                desired_signal = SIZE_BASE
        
        # SHORT entries (only when HTF is bearish)
        elif htf_1w_bear and htf_1d_bear:
            if fisher_overbought and vol_spike:
                # Primary: Fisher overbought + vol spike = rally exhaustion
                desired_signal = -SIZE_STRONG
            elif fisher_cross_short and rsi_14[i] > 50:
                # Continuation: Fisher cross + RSI not oversold
                desired_signal = -SIZE_BASE
            elif fisher[i] > 0.5 and rsi_14[i] > 60:
                # Rally into resistance in downtrend
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