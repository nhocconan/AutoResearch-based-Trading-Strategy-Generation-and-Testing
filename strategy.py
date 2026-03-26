#!/usr/bin/env python3
"""
Experiment #006: 4h Williams Alligator + Vortex + Volume + Choppiness

Hypothesis: Williams Alligator (Jaw/Teeth/Lips) provides superior trend structure 
vs Donchian/KAMA. Vortex confirms momentum direction. Volume spike validates 
breakouts. Choppiness filters ranging markets.

Why this should work in BOTH bull AND bear:
- Alligator self-adjusts to volatility (widens in chop, tightens in trends)
- Vortex works equally in both directions (momentum-based, not directional bias)
- Volume spike confirmation catches institutional moves in both directions
- 4h captures medium-term swings without overtrading

Target: 75-150 total train trades (tight entry), Sharpe>0.6
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_alligator_vortex_volume_chop_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_alligator(high, low, period=13):
    """
    Williams Alligator - 3 smoothed moving averages
    Jaw (blue): SMMA of median price, period 13
    Teeth (red): SMMA of median price, period 8
    Lips (green): SMMA of median price, period 5
    """
    n = len(high)
    if n < period + 5:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    median = (high + low) / 2.0
    
    def smma(series, period):
        result = np.full(len(series), np.nan, dtype=np.float64)
        smma_val = series[0]
        result[0] = smma_val
        for i in range(1, len(series)):
            smma_val = smma_val + (series[i] - smma_val) / period
            result[i] = smma_val
        return result
    
    jaw = smma(median, 13)
    teeth = smma(median, 8)
    lips = smma(median, 5)
    
    return jaw, teeth, lips

def calculate_vortex(high, low, close, period=14):
    """
    Vortex Indicator - identifies trend reversal
    VM+ = |High - Low(-1)|, VM- = |Low - High(-1)|
    VI+ = EMA(VM+, period) / EMA(TR, period)
    VI- = EMA(VM-, period) / EMA(TR, period)
    """
    n = len(high)
    if n < period + 2:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    high_low = high - low
    low_shift = np.roll(low, 1)
    low_shift[0] = low[0]
    high_shift = np.roll(high, 1)
    high_shift[0] = high[0]
    
    vm_plus = np.abs(high - low_shift)
    vm_minus = np.abs(low - high_shift)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], 
                   abs(high[i] - close[i-1]), 
                   abs(low[i] - close[i-1]))
    
    vi_plus = pd.Series(vm_plus).ewm(span=period, min_periods=period, adjust=False).mean().values
    vi_minus = pd.Series(vm_minus).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_ema = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    vi_plus_norm = np.zeros(n)
    vi_minus_norm = np.zeros(n)
    for i in range(n):
        if tr_ema[i] > 1e-10:
            vi_plus_norm[i] = vi_plus[i] / tr_ema[i]
            vi_minus_norm[i] = vi_minus[i] / tr_ema[i]
    
    return vi_plus_norm, vi_minus_norm

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

def calculate_volume_ratio(volume, period=20):
    """Volume ratio vs moving average - spike detection"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    ratio = np.zeros(n)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            ratio[i] = volume[i] / vol_ma[i]
    return ratio

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - trend vs range"""
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - for structure confirmation"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values if "volume" in prices.columns else np.ones(len(close))
    n = len(close)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # HTF HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=48)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    jaw, teeth, lips = calculate_alligator(high, low, period=13)
    vi_plus, vi_minus = calculate_vortex(high, low, close, period=14)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup
    min_bars = 60
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(jaw[i]) or np.isnan(vi_plus[i]) or np.isnan(chop_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(donch_upper[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # Regime: trend only (CHOP < 50) - avoid ranging markets
        is_trending = chop_14[i] < 50.0
        
        # Alligator state
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        
        # Bullish: price above all lines, lines aligned up
        alligator_bullish = (close[i] > jaw_val and close[i] > teeth_val and close[i] > lips_val and
                             lips_val > teeth_val and teeth_val > jaw_val)
        
        # Bearish: price below all lines, lines aligned down
        alligator_bearish = (close[i] < jaw_val and close[i] < teeth_val and close[i] < lips_val and
                            lips_val < teeth_val and teeth_val < jaw_val)
        
        # Vortex confirmation
        vi_bullish = vi_plus[i] > vi_minus[i]
        vi_bearish = vi_minus[i] > vi_plus[i]
        
        # Volume spike (>1.5x average)
        vol_spike = vol_ratio[i] > 1.5 if not np.isnan(vol_ratio[i]) else False
        
        # 1d HTF trend bias
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # Donchian breakout confirmation
        donch_break_long = close[i] > donch_upper[i-1] if i > 0 and not np.isnan(donch_upper[i-1]) else False
        donch_break_short = close[i] < donch_lower[i-1] if i > 0 and not np.isnan(donch_lower[i-1]) else False
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if is_trending:
            # LONG: Alligator bullish + Vortex bullish + HTF bias + Volume spike
            if alligator_bullish and vi_bullish and price_above_1d and vol_spike:
                desired_signal = SIZE
            
            # SHORT: Alligator bearish + Vortex bearish + HTF bias + Volume spike
            elif alligator_bearish and vi_bearish and price_below_1d and vol_spike:
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK (3x ATR) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 3.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 3.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE ===
        if desired_signal >= SIZE * 0.9:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.9:
            final_signal = -SIZE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = close[i] - 3.0 * entry_atr
                else:
                    stop_price = close[i] + 3.0 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals