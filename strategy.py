#!/usr/bin/env python3
"""
Experiment #004: 1d Williams Alligator + Fractals + Weekly Trend Filter

HYPOTHESIS: Williams Alligator (SMMA 5/8/13) captures institutional trend 
direction. Fractals mark key breakout/breakdown points. Weekly HMA confirms 
the broader trend direction. Alligator "awake" (lines spread apart) = strong 
trend, "sleeping" = chop/no trade.

WHY THIS SHOULD WORK IN BOTH BULL AND BEAR:
- Bull markets: Long when weekly uptrend + bullish alligator + fractal break
- Bear markets: Short when weekly downtrend + bearish alligator + fractal break
- Range: Alligator sleeping = no trades (correctly skips chop)

TARGET: 50-100 total trades over 4 years (12-25/year on 1d).
This is achievable since we have ~250 bars/year on 1d.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_williams_alligator_weekly_v1"
timeframe = "1d"
leverage = 1.0

def calculate_smma(close, period):
    """Smoothed Moving Average (Williams Alligator component)"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    result = np.full(n, np.nan, dtype=np.float64)
    result[period - 1] = np.nanmean(close[:period])
    
    for i in range(period, n):
        if not np.isnan(result[i - 1]) and not np.isnan(close[i]):
            result[i] = (result[i - 1] * (period - 1) + close[i]) / period
    
    return result

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

def calculate_fractals(high, low, period=5):
    """Williams Fractals - local extrema marked N bars after formation"""
    n = len(high)
    fractal_up = np.full(n, np.nan, dtype=np.float64)
    fractal_down = np.full(n, np.nan, dtype=np.float64)
    
    # Fractal forms when bar i is center of 5-bar pattern
    # We mark it period bars later (fractal "confirmation")
    for i in range(period, n - period):
        # Bullish fractal: highest high at center, lower highs on sides
        is_bull = True
        for j in range(i - period, i + period + 1):
            if j != i and high[j] >= high[i]:
                is_bull = False
                break
        if is_bull:
            # Mark fractal at current bar (will be "visible" after period bars)
            fractal_up[i] = high[i]
        
        # Bearish fractal: lowest low at center, higher lows on sides
        is_bear = True
        for j in range(i - period, i + period + 1):
            if j != i and low[j] <= low[i]:
                is_bear = False
                break
        if is_bear:
            fractal_down[i] = low[i]
    
    return fractal_up, fractal_down

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly HMA for trend direction
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=13)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Daily Williams Alligator components
    jaw = calculate_smma(close, 13)   # Jaw = blue = slowest
    teeth = calculate_smma(close, 8)  # Teeth = red = medium
    lips = calculate_smma(close, 5)   # Lips = green = fastest
    
    # ATR for stoploss
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Calculate fractals
    fractal_up, fractal_down = calculate_fractals(high, low, period=5)
    
    # Volume SMA for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_sma > 0, vol_sma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Trade cooldown (bars since last exit)
    bars_since_exit = 999
    
    # Warmup: need jaw(13), ATR(14), vol_sma(20), fractals(5)
    warmup = 60
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === WEEKLY TREND FILTER ===
        weekly_uptrend = close[i] > hma_1w_aligned[i]
        weekly_downtrend = close[i] < hma_1w_aligned[i]
        
        # === ALLIGATOR STATE ===
        # Bullish: lips > teeth > jaw (lines spread, pointing up)
        bull_alligator = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
        # Bearish: lips < teeth < jaw (lines spread, pointing down)
        bear_alligator = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])
        
        # Alligator spread (trend strength)
        jaw_teeth_spread = abs(teeth[i] - jaw[i]) / jaw[i] if jaw[i] != 0 else 0
        teeth_lips_spread = abs(lips[i] - teeth[i]) / teeth[i] if teeth[i] != 0 else 0
        total_spread = jaw_teeth_spread + teeth_lips_spread
        
        # Alligator "awake" when spread > 0.3% (not sleeping)
        alligator_awake = total_spread > 0.003
        
        # === FRACTAL CHECK ===
        # Current price vs recent fractals
        bull_fractal_price = fractal_up[i] if not np.isnan(fractal_up[i]) else 0
        bear_fractal_price = fractal_down[i] if not np.isnan(fractal_down[i]) else 0
        
        # Price broke above recent bullish fractal
        bull_fractal_broken = (bull_fractal_price > 0 and 
                               close[i] > bull_fractal_price and
                               high[i] > bull_fractal_price)
        
        # Price broke below recent bearish fractal
        bear_fractal_broken = (bear_fractal_price > 0 and 
                               close[i] < bear_fractal_price and
                               low[i] < bear_fractal_price)
        
        # === VOLUME CONFIRMATION ===
        vol_confirm = vol_ratio[i] > 1.3
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: Weekly uptrend + bullish alligator + fractal break + volume
        if weekly_uptrend and bull_alligator and alligator_awake:
            if bull_fractal_broken:
                if vol_confirm:
                    desired_signal = SIZE
                else:
                    desired_signal = SIZE * 0.5  # Partial without volume
        
        # SHORT: Weekly downtrend + bearish alligator + fractal break + volume
        if weekly_downtrend and bear_alligator and alligator_awake:
            if bear_fractal_broken:
                if vol_confirm:
                    desired_signal = -SIZE
                else:
                    desired_signal = -SIZE * 0.5  # Partial without volume
        
        # === COOLDOWN CHECK ===
        if bars_since_exit < 10:
            if in_position:
                # Keep existing position
                pass
            else:
                # No new entries during cooldown
                desired_signal = 0.0
        
        # === STOPLOSS CHECK ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
            bars_since_exit = 0
        
        # === TAKE PROFIT (alligator reversal) ===
        tp_triggered = False
        
        if in_position and position_side > 0:
            # Exit long when alligator starts closing
            if lips[i] < teeth[i] or teeth[i] < jaw[i]:
                tp_triggered = True
        
        if in_position and position_side < 0:
            # Exit short when alligator starts closing
            if lips[i] > teeth[i] or teeth[i] > jaw[i]:
                tp_triggered = True
        
        if tp_triggered:
            desired_signal = 0.0
            bars_since_exit = 0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New entry or reversal
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
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
        
        # Update cooldown counter
        if not in_position:
            bars_since_exit += 1
        
        signals[i] = desired_signal
    
    return signals