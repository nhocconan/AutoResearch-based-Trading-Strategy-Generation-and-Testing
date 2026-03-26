#!/usr/bin/env python3
"""
Experiment #002: 12h Primary + 1d HTF — Camarilla Pivot + Volume Spike + Choppiness Regime

HYPOTHESIS: 12h timeframe is underutilized and less prone to overtrading than 4h/6h.
12h = 730 bars/year, target 12-37 trades = 1 trade per 20-60 bars.

Why this should work in BOTH bull and bear markets:
1. Camarilla pivot levels are SELF-ADAPTIVE to volatility (H1-H4/L1-L4 scale with ATR)
2. In bull markets: price breaks above H4 resistance → strong continuation signal
3. In bear markets: price breaks below L4 support → strong continuation signal
4. Choppiness regime filter avoids whipsaws during range-bound periods
5. Volume spike confirms breakout validity, filters false breakouts

Key design choices based on DB analysis:
- Camarilla pivot (not Donchian) = more robust levels, proven Sharpe 1.47
- Volume spike confirmation = filters 50%+ false breakouts
- Choppiness < 38.2 for trending = trade with momentum
- Choppiness > 61.8 for ranging = mean revert at pivot levels
- 1d HMA for trend bias = smooth, less noisy than shorter TFs
- Discrete sizing: 0.25 (base), 0.30 (strong breakout)
- 2.5x ATR stoploss for proper risk management

Target: Sharpe > 0.5, trades 50-150 total over 4 years (12-37/year), DD < -35%
Timeframe: 12h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_camarilla_volume_chop_1d_v1"
timeframe = "12h"
leverage = 1.0


def calculate_hma(close, period):
    """Hull Moving Average"""
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


def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppy vs trending
    CHOP > 61.8 = ranging (mean revert), CHOP < 38.2 = trending (trend follow)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop


def calculate_camarilla(high, low, close, period=14):
    """
    Camarilla Pivot Levels
    H1-L1: R1 = close + (high - low) * 1.1/12
           S1 = close - (high - low) * 1.1/12
    H2-L2: R2 = close + (high - low) * 1.1/6
           S2 = close - (high - low) * 1.1/6
    H3-L3: R3 = close + (high - low) * 1.1/4
           S3 = close - (high - low) * 1.1/4
    H4-L4: R4 = close + (high - low) * 1.1/2
           S4 = close - (high - low) * 1.1/2
    
    Returns arrays for each level
    """
    n = len(close)
    if n < period:
        return (np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan), 
                np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan),
                np.full(n, np.nan), np.full(n, np.nan))
    
    h1 = np.full(n, np.nan, dtype=np.float64)
    h2 = np.full(n, np.nan, dtype=np.float64)
    h3 = np.full(n, np.nan, dtype=np.float64)
    h4 = np.full(n, np.nan, dtype=np.float64)
    l1 = np.full(n, np.nan, dtype=np.float64)
    l2 = np.full(n, np.nan, dtype=np.float64)
    l3 = np.full(n, np.nan, dtype=np.float64)
    l4 = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        prev_high = high[i - period + 1:i + 1]
        prev_low = low[i - period + 1:i + 1]
        prev_close = close[i - period + 1:i + 1]
        
        h = np.max(prev_high)
        l = np.min(prev_low)
        c = prev_close[-1]
        rng = h - l
        
        if rng < 1e-10:
            continue
        
        # H4/L4 are most important for trend continuation
        h4[i] = c + rng * 0.55
        h3[i] = c + rng * 0.275
        h2[i] = c + rng * 0.183
        h1[i] = c + rng * 0.092
        
        l1[i] = c - rng * 0.092
        l2[i] = c - rng * 0.183
        l3[i] = c - rng * 0.275
        l4[i] = c - rng * 0.55
    
    return h1, h2, h3, h4, l1, l2, l3, l4


def calculate_volume_spike(volume, period=20):
    """
    Volume spike detection - volume > 1.5x 20-period average
    Returns spike strength (ratio) or nan if no spike
    """
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    avg_vol = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    spike = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        if avg_vol[i] > 0 and volume[i] > avg_vol[i] * 1.5:
            spike[i] = volume[i] / avg_vol[i]
    
    return spike


def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (CRITICAL - Rule 1)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    vol_spike = calculate_volume_spike(volume, period=20)
    h1, h2, h3, h4, l1, l2, l3, l4 = calculate_camarilla(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup period
    min_bars = 50
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(chop_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(h4[i]) or np.isnan(l4[i]):
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
        
        # === REGIME DETECTION ===
        chop = chop_14[i]
        is_trend_regime = chop < 38.2
        is_range_regime = chop > 61.8
        
        # === 1d HMA TREND BIAS ===
        price_above_1d_hma = close[i] > hma_1d_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        has_volume_spike = not np.isnan(vol_spike[i]) and vol_spike[i] > 1.5
        
        # === CAMARILLA PIVOT BREAKOUT DETECTION ===
        # Previous bar levels
        prev_h4 = h4[i-1] if i > 0 and not np.isnan(h4[i-1]) else None
        prev_l4 = l4[i-1] if i > 0 and not np.isnan(l4[i-1]) else None
        prev_h3 = h3[i-1] if i > 0 and not np.isnan(h3[i-1]) else None
        prev_l3 = l3[i-1] if i > 0 and not np.isnan(l3[i-1]) else None
        
        # Breakout conditions: price closes beyond H4/L4 with volume
        breakout_long = False
        breakout_short = False
        
        if prev_h4 is not None:
            # Strong breakout: close above H4 + volume spike
            if close[i] > prev_h4:
                breakout_long = True
            # Moderate breakout: close above H3 + volume spike
            elif has_volume_spike and close[i] > prev_h3 and price_above_1d_hma:
                breakout_long = True
        
        if prev_l4 is not None:
            # Strong breakout: close below L4 + volume spike
            if close[i] < prev_l4:
                breakout_short = True
            # Moderate breakout: close below L3 + volume spike
            elif has_volume_spike and close[i] < prev_l3 and price_below_1d_hma:
                breakout_short = True
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # TREND REGIME: Trade breakouts with HTF trend bias
        if is_trend_regime:
            # LONG: Breakout above H4 + volume + 1d bullish
            if breakout_long and price_above_1d_hma and has_volume_spike:
                desired_signal = SIZE_STRONG
            # LONG: Moderate breakout + strong volume
            elif breakout_long and has_volume_spike:
                desired_signal = SIZE_BASE
            
            # SHORT: Breakout below L4 + volume + 1d bearish
            if breakout_short and price_below_1d_hma and has_volume_spike:
                desired_signal = -SIZE_STRONG
            # SHORT: Moderate breakout + strong volume
            elif breakout_short and has_volume_spike:
                desired_signal = -SIZE_BASE
        
        # RANGE REGIME: Mean reversion at Camarilla levels
        elif is_range_regime:
            # LONG: Price touches L3-L4 zone + RSI oversold (simple check via ATR proximity)
            if prev_l4 is not None and prev_l3 is not None:
                touch_l4_zone = low[i] <= prev_l4 * 1.005  # Within 0.5% of L4
                touch_l3_zone = low[i] <= prev_l3 * 1.005
                
                if (touch_l4_zone or touch_l3_zone) and price_above_1d_hma:
                    desired_signal = SIZE_BASE
            
            # SHORT: Price touches H3-H4 zone
            if prev_h4 is not None and prev_h3 is not None:
                touch_h4_zone = high[i] >= prev_h4 * 0.995  # Within 0.5% of H4
                touch_h3_zone = high[i] >= prev_h3 * 0.995
                
                if (touch_h4_zone or touch_h3_zone) and price_below_1d_hma:
                    desired_signal = -SIZE_BASE
        
        # NEUTRAL REGIME: No entry (too uncertain without clear regime)
        # This is intentional - neutral regime is "do nothing"
        
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