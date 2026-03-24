#!/usr/bin/env python3
"""
Experiment #571: 6h Primary + 1d/1w HTF — Fisher Transform Reversal + HTF Trend Filter

Hypothesis: Ehlers Fisher Transform excels at catching reversals in bear/range markets
(2022 crash, 2025 bear) where simple trend strategies fail. Fisher normalizes price
to Gaussian distribution, making extremes statistically significant. Combined with
1d/1w HMA for macro trend filter, this should:
1. Generate sufficient trades (Fisher crosses happen regularly)
2. Filter counter-trend trades (HTF alignment)
3. Work in both bull and bear regimes (reversal + trend following)
4. Outperform CRSI/Choppiness strategies that have failed 40+ times

Key differences from failed strategies:
1. Fisher Transform instead of RSI/CRSI (better reversal detection)
2. Volume confirmation (taker_buy_volume ratio) for entry strength
3. Adaptive sizing based on Fisher extreme level (deeper extreme = larger size)
4. ATR-based trailing stop (2.0x ATR)
5. Simpler regime logic (Fisher value itself indicates regime)

Strategy logic:
1. 1w HMA(21) = macro trend bias
2. 1d HMA(21) = medium trend bias  
3. 6h Fisher(9) = entry timing (crosses -1.5/+1.5)
4. 6h Volume ratio = confirmation (taker_buy > 55% for long)
5. 6h ATR(14) = stoploss and position sizing
6. HTF alignment required for full size, partial size without

Target: Sharpe>0.40, trades>=80 train (20/year), trades>=10 test
Timeframe: 6h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_hma_volume_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform
    Normalizes price to Gaussian distribution for statistically significant extremes
    
    Steps:
    1. Calculate typical price = (high + low + close) / 3
    2. Normalize to -1 to +1 range using highest high / lowest low over period
    3. Apply Fisher transform: 0.5 * ln((1+x)/(1-x))
    4. Smooth with EMA
    
    Entry signals:
    - Long: Fisher crosses above -1.5 from below
    - Short: Fisher crosses below +1.5 from above
    """
    n = len(close)
    if n < period + 5:
        return np.full(n, np.nan)
    
    fisher = np.zeros(n)
    fisher[:] = np.nan
    
    # Calculate typical price
    typical = (high + low + close) / 3.0
    
    # Normalize price to -1 to +1
    normalized = np.zeros(n)
    normalized[:] = np.nan
    
    for i in range(period, n):
        highest = np.nanmax(high[i-period+1:i+1])
        lowest = np.nanmin(low[i-period+1:i+1])
        price_range = highest - lowest
        
        if price_range > 1e-10:
            # Normalize to 0-1, then scale to -0.99 to +0.99 (avoid division by zero)
            norm = 2.0 * (typical[i] - lowest) / price_range - 1.0
            normalized[i] = max(-0.99, min(0.99, norm))
        else:
            normalized[i] = 0.0
    
    # Apply Fisher transform
    for i in range(period, n):
        if abs(normalized[i]) < 0.99:
            fisher_raw = 0.5 * np.log((1.0 + normalized[i]) / (1.0 - normalized[i]))
            fisher[i] = fisher_raw
        else:
            fisher[i] = fisher[i-1] if i > period else 0.0
    
    # Smooth with EMA (period=3)
    fisher_smooth = pd.Series(fisher).ewm(span=3, min_periods=3, adjust=False).mean().values
    
    return fisher_smooth

def calculate_volume_ratio(taker_buy_volume, volume):
    """
    Calculate taker buy volume ratio
    Ratio > 0.55 = buying pressure
    Ratio < 0.45 = selling pressure
    """
    n = len(volume)
    ratio = np.zeros(n)
    ratio[:] = np.nan
    
    for i in range(n):
        if volume[i] > 1e-10:
            ratio[i] = taker_buy_volume[i] / volume[i]
        else:
            ratio[i] = 0.5
    
    return ratio

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

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

def calculate_rsi(close, period=14):
    """Relative Strength Index for additional filter"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_volume = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1d HMA for medium trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for macro trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 6h indicators
    fisher = calculate_fisher_transform(high, low, close, period=9)
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    vol_ratio = calculate_volume_ratio(taker_buy_volume, volume)
    
    signals = np.zeros(n)
    
    # Position sizing levels (discrete to minimize fee churn)
    SIZE_FULL = 0.30
    SIZE_HALF = 0.20
    SIZE_QUARTER = 0.10
    
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
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1w macro + 1d medium) ===
        htf_bull = close[i] > hma_1d_aligned[i] and close[i] > hma_1w_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i] and close[i] < hma_1w_aligned[i]
        htf_neutral = not htf_bull and not htf_bear
        
        # === FISHER TRANSFORM SIGNALS ===
        # Fisher crosses above -1.5 from below = long signal
        fisher_cross_long = (fisher[i] > -1.5 and fisher[i-1] <= -1.5) if i > 0 else False
        # Fisher crosses below +1.5 from above = short signal
        fisher_cross_short = (fisher[i] < 1.5 and fisher[i-1] >= 1.5) if i > 0 else False
        
        # Fisher extreme levels (deeper = stronger signal)
        fisher_deep_oversold = fisher[i] < -2.0
        fisher_deep_overbought = fisher[i] > 2.0
        fisher_moderate_oversold = fisher[i] < -1.0 and fisher[i] > -2.0
        fisher_moderate_overbought = fisher[i] > 1.0 and fisher[i] < 2.0
        
        # Fisher turning (momentum change)
        fisher_turning_long = fisher[i] > fisher[i-1] if i > 0 else False
        fisher_turning_short = fisher[i] < fisher[i-1] if i > 0 else False
        
        # === VOLUME CONFIRMATION ===
        vol_buying = vol_ratio[i] > 0.55
        vol_selling = vol_ratio[i] < 0.45
        vol_neutral = not vol_buying and not vol_selling
        
        # === RSI FILTER (avoid extreme counter-trend) ===
        rsi_oversold = rsi[i] < 40.0
        rsi_overbought = rsi[i] > 60.0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        signal_strength = 0.0
        
        # LONG ENTRIES
        if fisher_cross_long or (fisher_deep_oversold and fisher_turning_long):
            # Full size: HTF bull + Fisher cross + volume confirmation
            if htf_bull and fisher_cross_long and vol_buying:
                desired_signal = SIZE_FULL
                signal_strength = 1.0
            # Half size: HTF bull + Fisher deep oversold
            elif htf_bull and fisher_deep_oversold:
                desired_signal = SIZE_HALF
                signal_strength = 0.7
            # Quarter size: HTF neutral + Fisher cross + RSI confirmation
            elif htf_neutral and fisher_cross_long and rsi_oversold:
                desired_signal = SIZE_QUARTER
                signal_strength = 0.5
            # Small long: HTF bear but Fisher extremely oversold (reversal play)
            elif htf_bear and fisher[i] < -2.5 and vol_buying:
                desired_signal = SIZE_QUARTER
                signal_strength = 0.4
        
        # SHORT ENTRIES
        elif fisher_cross_short or (fisher_deep_overbought and fisher_turning_short):
            # Full size: HTF bear + Fisher cross + volume confirmation
            if htf_bear and fisher_cross_short and vol_selling:
                desired_signal = -SIZE_FULL
                signal_strength = 1.0
            # Half size: HTF bear + Fisher deep overbought
            elif htf_bear and fisher_deep_overbought:
                desired_signal = -SIZE_HALF
                signal_strength = 0.7
            # Quarter size: HTF neutral + Fisher cross + RSI confirmation
            elif htf_neutral and fisher_cross_short and rsi_overbought:
                desired_signal = -SIZE_QUARTER
                signal_strength = 0.5
            # Small short: HTF bull but Fisher extremely overbought (reversal play)
            elif htf_bull and fisher[i] > 2.5 and vol_selling:
                desired_signal = -SIZE_QUARTER
                signal_strength = 0.4
        
        # === STOPLOSS CHECK (2.0x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            if low[i] < stop_price:
                stoploss_triggered = True
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            if high[i] > stop_price:
                stoploss_triggered = True
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
        
        if stoploss_triggered:
            desired_signal = 0.0
            signal_strength = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        final_signal = 0.0
        if desired_signal > 0:
            if signal_strength >= 0.9:
                final_signal = SIZE_FULL
            elif signal_strength >= 0.6:
                final_signal = SIZE_HALF
            elif signal_strength >= 0.3:
                final_signal = SIZE_QUARTER
        elif desired_signal < 0:
            if signal_strength >= 0.9:
                final_signal = -SIZE_FULL
            elif signal_strength >= 0.6:
                final_signal = -SIZE_HALF
            elif signal_strength >= 0.3:
                final_signal = -SIZE_QUARTER
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
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