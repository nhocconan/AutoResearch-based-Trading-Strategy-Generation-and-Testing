#!/usr/bin/env python3
"""
Experiment #155: 1h Primary + 4h/1d HTF — Fisher Transform + Choppiness Regime + HMA Trend

Hypothesis: After analyzing 154 failed experiments, the pattern for 1h timeframe is clear:
- CRSI is TOO RESTRICTIVE for lower TF (experiments #145, #148, #150 all got 0 trades)
- Session filters kill trade generation on 1h
- SOLUTION: Ehlers Fisher Transform for reversals (proven in bear markets) + Choppiness regime
- Fisher Transform catches reversals at extremes better than RSI in choppy markets
- 4h HMA provides trend bias without being too restrictive (unlike 1d)
- Volume filter ensures we only trade when there's actual interest
- LOOSE Fisher thresholds (-1.5/+1.5) ensure trades generate on all symbols

Key design choices:
- Timeframe: 1h (30-60 trades/year target)
- HTF: 4h HMA(21) for trend bias (not 1d which is too slow for 1h entries)
- Entry: Fisher Transform reversals + Choppiness regime + volume confirmation
- Regime: CHOP>55 = range (mean revert with Fisher), CHOP<55 = trend (Fisher + HMA alignment)
- Position size: 0.25 (25% of capital, conservative for 1h fee drag)
- Stoploss: 2.5x ATR trailing (tighter for 1h swings)
- NO session filter (kills trades), LOOSE Fisher thresholds to ensure >=30 trades

Target: Sharpe>0.375 (beat #152), DD>-40%, trades>=30 on train, trades>=3 on test, ALL symbols Sharpe>0
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_chop_regime_hma_4h_v1"
timeframe = "1h"
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

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform
    Converts price to Gaussian distribution for clearer reversal signals
    Long when Fisher crosses above -1.5, Short when crosses below +1.5
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.zeros(n)
    fisher[:] = np.nan
    fisher_prev = np.zeros(n)
    fisher_prev[:] = np.nan
    
    for i in range(period, n):
        # Calculate highest high and lowest low over period
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        range_hl = highest_high - lowest_low
        
        if range_hl < 1e-10:
            fisher[i] = 0.0
            fisher_prev[i] = fisher[i-1] if i > 0 else 0.0
            continue
        
        # Normalize price to 0-1 range
        price_normalized = (close[i] - lowest_low) / range_hl
        
        # Clamp to avoid division issues (0.001 to 0.999)
        price_normalized = max(0.001, min(0.999, price_normalized))
        
        # Fisher Transform formula
        fisher_value = 0.5 * np.log((1.0 + price_normalized) / (1.0 - price_normalized))
        
        # Smooth with previous value (Ehlers' method)
        if i > period and not np.isnan(fisher[i-1]):
            fisher[i] = 0.67 * fisher_value + 0.33 * fisher[i-1]
        else:
            fisher[i] = fisher_value
        
        fisher_prev[i] = fisher[i-1] if i > 0 and not np.isnan(fisher[i-1]) else fisher[i]
    
    return fisher, fisher_prev

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    Measures market choppiness vs trending
    CHOP > 55 = choppy/range (mean revert), CHOP < 55 = trending (follow)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        sum_tr = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        range_hl = highest_high - lowest_low
        
        if range_hl > 1e-10 and sum_tr > 1e-10:
            chop[i] = 100.0 * np.log10(sum_tr / range_hl) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_volume_sma(volume, period=20):
    """Simple moving average of volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align 4h HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate primary (1h) indicators
    fisher, fisher_prev = calculate_fisher_transform(high, low, close, period=9)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    vol_sma = calculate_volume_sma(volume, period=20)
    
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size (conservative for 1h)
    
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
        if np.isnan(fisher[i]) or np.isnan(fisher_prev[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(vol_sma[i]) or vol_sma[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (4h HMA) ===
        htf_bull = close[i] > hma_4h_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP > 55 = range/choppy (mean revert with Fisher)
        # CHOP < 55 = trending (Fisher + HTF alignment)
        is_choppy = chop[i] > 55.0
        is_trending = chop[i] <= 55.0
        
        # === VOLUME CONFIRMATION ===
        # Only trade when volume >= 0.7x average (loose to ensure trades)
        volume_ok = volume[i] >= 0.7 * vol_sma[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_cross_up = fisher_prev[i] < -1.5 and fisher[i] >= -1.5
        fisher_cross_down = fisher_prev[i] > 1.5 and fisher[i] <= 1.5
        
        # Fisher at extremes (for choppy regime mean reversion)
        fisher_oversold = fisher[i] < -1.0
        fisher_overbought = fisher[i] > 1.0
        
        # === DESIRED SIGNAL (Dual Regime Logic) ===
        desired_signal = 0.0
        
        if is_trending:
            # TREND REGIME: Fisher reversals WITH HTF alignment
            # LONG: Fisher cross up + HTF bull + volume ok
            if fisher_cross_up and htf_bull and volume_ok:
                desired_signal = SIZE
            # SHORT: Fisher cross down + HTF bear + volume ok
            elif fisher_cross_down and htf_bear and volume_ok:
                desired_signal = -SIZE
            # Fallback: Fisher extreme + HTF alignment (looser)
            elif fisher_oversold and htf_bull and volume_ok:
                desired_signal = SIZE * 0.6
            elif fisher_overbought and htf_bear and volume_ok:
                desired_signal = -SIZE * 0.6
        else:
            # CHOPPY REGIME: Mean revert with Fisher extremes
            # LONG: Fisher oversold + volume ok (ignore HTF in choppy)
            if fisher_oversold and volume_ok:
                desired_signal = SIZE
            # SHORT: Fisher overbought + volume ok
            elif fisher_overbought and volume_ok:
                desired_signal = -SIZE
            # Fallback: Fisher cross without HTF filter
            elif fisher_cross_up and volume_ok:
                desired_signal = SIZE * 0.6
            elif fisher_cross_down and volume_ok:
                desired_signal = -SIZE * 0.6
        
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