#!/usr/bin/env python3
"""
Experiment #053: 1d Primary + 1w HTF — Fisher Transform + Vol Regime + BB Mean Reversion

Hypothesis: Ehlers Fisher Transform excels at catching reversals in bear/range markets
(2022 crash, 2025 bear) while Bollinger Band extremes + volatility spikes confirm
oversold/overbought conditions. Combined with 1w HMA for major trend bias, this should:
1. Work in 2021 bull (trend following with Fisher confirmation)
2. Work in 2022 crash (Fisher catches reversal bottoms)
3. Work in 2025 bear (mean reversion at BB extremes)
4. Generate 20-50 trades/year on 1d (low fee drag)

Key components:
- Fisher Transform (period=9): Normalizes price to -1.5 to +1.5 range, catches reversals
- Volatility Regime: ATR(7)/ATR(21) ratio detects extreme moves (>2.0 = panic/euphoria)
- Bollinger Bands (20, 2.5): Wide bands for extreme mean reversion entries
- 1w HMA(21): Major trend bias (don't short if weekly HMA strongly bull)
- Asymmetric sizing: 0.30 for high-confidence, 0.20 for moderate

Target: Sharpe>0.351, trades>30/symbol train, >3/symbol test, DD>-40%
Timeframe: 1d (target 20-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_fisher_bb_volregime_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - less lag than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_p = int(np.sqrt(period))
    
    def wma(data, span):
        res = np.full(len(data), np.nan)
        w = np.arange(1, span + 1, dtype=np.float64)
        for i in range(span - 1, len(data)):
            res[i] = np.sum(data[i - span + 1:i + 1] * w) / np.sum(w)
        return res
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    double_wma = 2.0 * wma_half - wma_full
    hma = wma(double_wma, sqrt_p)
    return hma

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform
    Normalizes price action to range approximately -1.5 to +1.5
    Catches reversals better than RSI in bear markets
    """
    n = len(high)
    if n < period:
        return np.full(n, np.nan)
    
    fisher = np.full(n, np.nan)
    trigger = np.full(n, np.nan)
    
    for i in range(period, n):
        # Calculate typical price
        hl2 = (high[i] + low[i]) / 2.0
        
        # Find highest high and lowest low over period
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        # Normalize to 0-1 range
        if highest > lowest:
            normalized = (hl2 - lowest) / (highest - lowest)
        else:
            normalized = 0.5
        
        # Clamp to avoid division issues
        normalized = max(0.001, min(0.999, normalized))
        
        # Fisher transform
        fisher_val = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
        
        # Smooth with previous value (Ehlers' method)
        if i > period and not np.isnan(fisher[i-1]):
            fisher[i] = 0.67 * fisher_val + 0.33 * fisher[i-1]
        else:
            fisher[i] = fisher_val
        
        # Trigger line (1-period lag)
        if i > period:
            trigger[i] = fisher[i-1]
    
    return fisher, trigger

def calculate_bollinger_bands(close, period=20, std_mult=2.5):
    """Bollinger Bands with wider std for extreme detection"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, lower, sma

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility and stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_vol_regime(atr, short_period=7, long_period=21):
    """
    Volatility Regime: ATR(short) / ATR(long)
    > 2.0 = volatility spike (panic/euphoria)
    < 0.8 = volatility compression (calm)
    """
    n = len(atr)
    if n < long_period:
        return np.full(n, np.nan)
    
    # Calculate short and long ATR
    tr = np.zeros(n)
    tr[0] = 0  # Will be overwritten
    for i in range(1, n):
        # We need high/low/close but only have ATR - approximate
        tr[i] = atr[i]  # Use ATR as proxy for TR
    
    atr_short = pd.Series(tr).ewm(span=short_period, min_periods=short_period, adjust=False).mean().values
    atr_long = pd.Series(tr).ewm(span=long_period, min_periods=long_period, adjust=False).mean().values
    
    vol_ratio = np.full(n, np.nan)
    for i in range(long_period, n):
        if atr_long[i] > 1e-10:
            vol_ratio[i] = atr_short[i] / atr_long[i]
    
    return vol_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for HTF trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    fisher, fisher_trigger = calculate_fisher_transform(high, low, period=9)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, period=20, std_mult=2.5)
    atr = calculate_atr(high, low, close, period=14)
    
    # Volatility regime (ATR ratio)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_7 = pd.Series(tr).ewm(span=7, min_periods=7, adjust=False).mean().values
    atr_21 = pd.Series(tr).ewm(span=21, min_periods=21, adjust=False).mean().values
    
    vol_ratio = np.full(n, np.nan)
    for i in range(21, n):
        if atr_21[i] > 1e-10:
            vol_ratio[i] = atr_7[i] / atr_21[i]
    
    signals = np.zeros(n)
    SIZE_HIGH = 0.30  # High confidence
    SIZE_MED = 0.20   # Medium confidence
    
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
        if np.isnan(fisher[i]) or np.isnan(fisher_trigger[i]):
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
        if np.isnan(vol_ratio[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1w HMA) ===
        hma_1w_bull = close[i] > hma_1w_aligned[i]
        hma_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        # Fisher crosses above -1.5 from below = bullish reversal
        fisher_bull_cross = (fisher[i] > -1.5 and fisher_trigger[i] <= -1.5)
        # Fisher crosses below +1.5 from above = bearish reversal
        fisher_bear_cross = (fisher[i] < 1.5 and fisher_trigger[i] >= 1.5)
        
        # Fisher extreme oversold/overbought
        fisher_oversold = fisher[i] < -1.2
        fisher_overbought = fisher[i] > 1.2
        
        # === BOLLINGER BAND POSITION ===
        bb_oversold = close[i] < bb_lower[i]
        bb_overbought = close[i] > bb_upper[i]
        
        # === VOLATILITY REGIME ===
        vol_spike = vol_ratio[i] > 2.0  # Panic/euphoria
        vol_calm = vol_ratio[i] < 0.8   # Compression
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        signal_confidence = 0  # 0=none, 1=med, 2=high
        
        # LONG SETUP: Fisher reversal + BB oversold + (vol spike OR weekly bull)
        if fisher_bull_cross or (fisher_oversold and bb_oversold):
            long_confidence = 0
            if vol_spike:
                long_confidence += 1  # Vol spike adds confidence
            if hma_1w_bull:
                long_confidence += 1  # Weekly trend adds confidence
            if bb_oversold:
                long_confidence += 1  # BB extreme adds confidence
            
            if long_confidence >= 2:
                desired_signal = SIZE_HIGH
                signal_confidence = 2
            elif long_confidence == 1:
                desired_signal = SIZE_MED
                signal_confidence = 1
        
        # SHORT SETUP: Fisher reversal + BB overbought + (vol spike OR weekly bear)
        if fisher_bear_cross or (fisher_overbought and bb_overbought):
            short_confidence = 0
            if vol_spike:
                short_confidence += 1
            if hma_1w_bear:
                short_confidence += 1
            if bb_overbought:
                short_confidence += 1
            
            if short_confidence >= 2:
                desired_signal = -SIZE_HIGH
                signal_confidence = 2
            elif short_confidence == 1:
                desired_signal = -SIZE_MED
                signal_confidence = 1
        
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
        if desired_signal >= SIZE_HIGH * 0.85:
            final_signal = SIZE_HIGH
        elif desired_signal >= SIZE_MED * 0.85:
            final_signal = SIZE_MED
        elif desired_signal <= -SIZE_HIGH * 0.85:
            final_signal = -SIZE_HIGH
        elif desired_signal <= -SIZE_MED * 0.85:
            final_signal = -SIZE_MED
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