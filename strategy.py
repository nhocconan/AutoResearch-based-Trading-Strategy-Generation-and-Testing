#!/usr/bin/env python3
"""
Experiment #043: Williams Alligator + Elder Ray + ADX Regime

HYPOTHESIS: Williams Alligator captures multi-timeframe trend structure using
fractal smoothing. Elder Ray measures institutional buying/selling pressure.
Combined with ADX regime filter, this captures 75-150 clean trades over 4 years.

WHY BOTH BULL AND BEAR:
- Bull market: Price > Alligator lines, lines aligned UP, Bull Power > 0 → Long
- Bear market: Price < Alligator lines, lines aligned DOWN, Bear Power < 0 → Short
- Range: Alligator lines compressed/horizontal → No trades (ADX filter)
- Alligator's 3 smoothed lines (Jaw, Teeth, Lips) naturally filter noise vs single EMA

DB REFERENCE: gen_pivot_breakout_vpin_proxy_sma200_regime_30m_v1 (Sharpe=1.787)
TARGET: 75-150 total trades over 4 years (19-37/year on 4h)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_alligator_elder_ray_adx_v1"
timeframe = "4h"
leverage = 1.0

def calculate_alligator(high, low, close, jaw_period=13, teeth_period=8, lips_period=5):
    """
    Williams Alligator indicator
    Jaw = SMMA(median, 13)
    Teeth = SMMA(median, 8)
    Lips = SMMA(median, 5)
    All lines above = bearish, below = bullish, compressed = neutral
    """
    n = len(close)
    median = (high + low) / 2.0
    
    def smma(series, period):
        result = np.full(n, np.nan, dtype=np.float64)
        if n < period:
            return result
        
        # Initialize with SMA
        cumsum = 0.0
        for i in range(period):
            cumsum += series[i]
        result[period - 1] = cumsum / period
        
        # SMMA formula
        prev = result[period - 1]
        for i in range(period, n):
            prev = (prev * (period - 1) + series[i]) / period
            result[i] = prev
        
        return result
    
    jaw = smma(median, jaw_period)
    teeth = smma(median, teeth_period)
    lips = smma(median, lips_period)
    
    return jaw, teeth, lips

def calculate_elder_ray(close, high, low, jaw_period=13, teeth_period=8):
    """
    Elder Ray - measures buying/selling pressure
    Bull Power = High - Alligator Teeth
    Bear Power = Low - Alligator Teeth
    """
    jaw, teeth, lips = calculate_alligator(high, low, close, jaw_period, teeth_period)
    
    bull_power = np.full(len(close), np.nan, dtype=np.float64)
    bear_power = np.full(len(close), np.nan, dtype=np.float64)
    
    for i in range(len(close)):
        if not np.isnan(teeth[i]):
            bull_power[i] = high[i] - teeth[i]
            bear_power[i] = low[i] - teeth[i]
    
    return bull_power, bear_power

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - measures trend strength"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    # Smooth with Wilder's method
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = np.zeros(n, dtype=np.float64)
    minus_di = np.zeros(n, dtype=np.float64)
    
    for i in range(period, n):
        plus_di[i] = 100 * np.mean(plus_dm[i-period+1:i+1]) / (atr[i] + 1e-10)
        minus_di[i] = 100 * np.mean(minus_dm[i-period+1:i+1]) / (atr[i] + 1e-10)
    
    dx = np.zeros(n, dtype=np.float64)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = np.full(n, np.nan, dtype=np.float64)
    adx[period + 13] = np.mean(dx[period:period + 14])
    for i in range(period + 14, n):
        adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    return adx

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 12h for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # 12h HMA(21) for trend direction
    def calc_hma(close, period):
        n = len(close)
        half = max(1, period // 2)
        sqrt_n = max(1, int(np.sqrt(period)))
        
        def wma(series, span):
            result = np.full(len(series), np.nan)
            weights = np.arange(1, span + 1, dtype=np.float64)
            for i in range(span - 1, len(series)):
                window = series[i - span + 1:i + 1]
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / np.sum(weights)
            return result
        
        wma_half = wma(close, half)
        wma_full = wma(close, period)
        
        diff = 2 * wma_half - wma_full
        return wma(diff, sqrt_n)
    
    hma_12h_raw = calc_hma(df_12h['close'].values, 21)
    hma_12h = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate 4h indicators
    jaw, teeth, lips = calculate_alligator(high, low, close)
    bull_power, bear_power = calculate_elder_ray(close, high, low)
    adx = calculate_adx(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 50
    
    for i in range(warmup, n):
        # Check indicators ready
        if np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]):
            signals[i] = 0.0
            position_side = 0
            continue
        
        if np.isnan(adx[i]) or np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            position_side = 0
            continue
        
        # 12h trend filter
        price_above_12h_hma = close[i] > hma_12h[i] if not np.isnan(hma_12h[i]) else True
        price_below_12h_hma = close[i] < hma_12h[i] if not np.isnan(hma_12h[i]) else False
        
        # ADX regime filter (trend must be strong enough)
        adx_strong = adx[i] > 22
        
        # Alligator spread (alignment of lines)
        alligator_bullish = (lips[i] > teeth[i] > jaw[i]) and (close[i] > lips[i])
        alligator_bearish = (lips[i] < teeth[i] < jaw[i]) and (close[i] < lips[i])
        
        # Elder Ray confirmation
        bull_strong = bull_power[i] > 0
        bear_strong = bear_power[i] < 0
        
        # Volume confirmation
        vol_confirm = vol_ratio[i] > 1.4
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if position_side == 0:
            # No position - look for entry
            
            # LONG: Alligator bullish + bull power positive + ADX regime + trend filter + vol
            if alligator_bullish and bull_strong and adx_strong and price_above_12h_hma:
                if vol_confirm:
                    desired_signal = SIZE
                else:
                    # Still enter without vol if other conditions very strong
                    if adx[i] > 28:
                        desired_signal = SIZE
            
            # SHORT: Alligator bearish + bear power negative + ADX regime + trend filter + vol
            if alligator_bearish and bear_strong and adx_strong and price_below_12h_hma:
                if vol_confirm:
                    desired_signal = -SIZE
                else:
                    if adx[i] > 28:
                        desired_signal = -SIZE
        
        elif position_side > 0:
            # Holding long - only exit on stoploss or reversal signal
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            
            # Stoploss check
            if low[i] < trailing_stop:
                desired_signal = 0.0
                position_side = 0
            else:
                # Check for reversal (bearish alligator + strong bear power)
                if alligator_bearish and bear_strong and adx_strong:
                    desired_signal = 0.0  # Exit long
                    position_side = 0
                else:
                    desired_signal = SIZE  # Hold long
        
        elif position_side < 0:
            # Holding short - only exit on stoploss or reversal signal
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            
            # Stoploss check
            if high[i] > trailing_stop:
                desired_signal = 0.0
                position_side = 0
            else:
                # Check for reversal (bullish alligator + strong bull power)
                if alligator_bullish and bull_strong and adx_strong:
                    desired_signal = 0.0  # Exit short
                    position_side = 0
                else:
                    desired_signal = -SIZE  # Hold short
        
        # === NEW ENTRY ===
        if desired_signal != 0.0 and (position_side == 0 or np.sign(desired_signal) != position_side):
            position_side = int(np.sign(desired_signal))
            entry_price = close[i]
            entry_atr = atr[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
        
        signals[i] = desired_signal
    
    return signals