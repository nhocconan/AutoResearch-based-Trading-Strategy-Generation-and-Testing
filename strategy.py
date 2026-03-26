#!/usr/bin/env python3
"""
Experiment #011: 6h Williams Alligator + ADX Trend Confirmation

HYPOTHESIS: Williams Alligator identifies institutional trend direction by 
smoothing price into three lines (jaw/teeth/lips). When all three align 
parallel (alligator "sleeping"), volatility is compressed. When they spread 
("awakening"), a trend begins. ADX confirms trend STRENGTH to filter false 
breakouts. This combination captures major trends while avoiding whipsaws 
during the 2022 bear market (when simple trend following failed).

Key insight from #002 (best session performer): Alligator wakeup alone works 
but generated 267 trades (too many). Adding ADX filter should:
1. Reduce trades by only allowing entries when ADX > threshold
2. Keep trades during strong trends, skip weak ones

TIMEFRAME: 6h primary
HTF: 1d for regime filter (SMA200 position)
TARGET: 75-200 total trades over 4 years (19-50/year)
SIZE: 0.25 (discrete)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_alligator_adx_1d_regime_v1"
timeframe = "6h"
leverage = 1.0

def calculate_alligator(high, low, close, jaw_period=13, teeth_period=8, lips_period=5):
    """
    Williams Alligator: Three smoothed moving averages
    - Jaw (blue): 13-period smoothed SMA, shifted 8 bars forward
    - Teeth (red): 8-period smoothed SMA, shifted 5 bars forward
    - Lips (green): 5-period smoothed SMA, shifted 3 bars forward
    """
    n = len(close)
    
    def smoothed(prices, period, shift):
        # SMMA (Smoothed Moving Average)
        result = np.full(n, np.nan, dtype=np.float64)
        if n < period:
            return result
        
        # First value is SMA
        smma = np.mean(prices[:period])
        result[period - 1] = smma
        
        # Subsequent values: (prev_smma * (period-1) + current) / period
        for i in range(period, n):
            smma = (smma * (period - 1) + prices[i]) / period
            result[i] = smma
        
        # Shift forward (positive = future shift in this context means looking ahead)
        # Williams Alligator shifts: jaw +8, teeth +5, lips +3
        shifted = np.full(n, np.nan, dtype=np.float64)
        for i in range(n - shift):
            shifted[i + shift] = result[i]
        return shifted
    
    jaw = smoothed((high + low) / 2, jaw_period, 8)
    teeth = smoothed((high + low) / 2, teeth_period, 5)
    lips = smoothed((high + low) / 2, lips_period, 3)
    
    return jaw, teeth, lips

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength without direction"""
    n = len(close)
    
    # True Range
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Directional Movement
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    # Smooth with EWM
    atr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # DX components
    plus_di = np.zeros(n, dtype=np.float64)
    minus_di = np.zeros(n, dtype=np.float64)
    dx = np.zeros(n, dtype=np.float64)
    
    for i in range(period, n):
        if atr_smooth[i] > 0:
            plus_di[i] = 100 * plus_dm_smooth[i] / atr_smooth[i]
            minus_di[i] = 100 * minus_dm_smooth[i] / atr_smooth[i]
            
            di_sum = plus_di[i] + minus_di[i]
            if di_sum > 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    # ADX is EWM of DX
    adx = pd.Series(dx).ewm(span=period, min_periods=period * 2, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss"""
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
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA200 for regime filter
    sma200_1d = df_1d['close'].values
    sma200_1d_raw = pd.Series(sma200_1d).rolling(window=200, min_periods=200).mean().values
    sma200_aligned = align_htf_to_ltf(prices, df_1d, sma200_1d_raw)
    
    # Calculate 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Williams Alligator
    jaw, teeth, lips = calculate_alligator(high, low, close)
    
    # ADX for trend strength
    adx, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE = 0.25  # Conservative sizing
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    bars_in_trade = 0
    
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(adx[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME (1d SMA200) ===
        price_above_1d_sma200 = close[i] > sma200_aligned[i] if not np.isnan(sma200_aligned[i]) else True
        
        # === ADX TREND STRENGTH ===
        adx_val = adx[i]
        strong_trend = adx_val > 25  # ADX above 25 = trending
        
        # === ALLIGATOR ALIGNMENT ===
        # Alligator "awake": all three lines spread in direction of trend
        # Bull trend: lips > teeth > jaw (green > red > blue going up)
        # Bear trend: lips < teeth < jaw (going down)
        
        alligator_bull = lips[i] > teeth[i] and teeth[i] > jaw[i]
        alligator_bear = lips[i] < teeth[i] and teeth[i] < jaw[i]
        alligator_sleeping = abs(lips[i] - jaw[i]) < 0.1 * jaw[i]  # Tight = sleeping
        
        # === DIRECTIONAL INDICATORS ===
        plus_di_val = plus_di[i] if not np.isnan(plus_di[i]) else 0
        minus_di_val = minus_di[i] if not np.isnan(minus_di[i]) else 0
        di_strength = plus_di_val - minus_di_val  # Positive = bullish pressure
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG ENTRY ===
            # Alligator bullish alignment + ADX confirms trend + price above SMA200
            if alligator_bull and strong_trend and price_above_1d_sma200:
                # Additional confirmation: +DI > -DI (bullish pressure)
                if plus_di_val > minus_di_val:
                    desired_signal = SIZE
            
            # === SHORT ENTRY ===
            # Alligator bearish alignment + ADX confirms trend + price below SMA200
            if alligator_bear and strong_trend and not price_above_1d_sma200:
                # Additional confirmation: -DI > +DI (bearish pressure)
                if minus_di_val > plus_di_val:
                    desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR trailing stop) ===
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
        
        # === EXIT CONDITIONS ===
        exit_triggered = False
        
        if in_position and position_side > 0:
            # Exit long: alligator turns bearish OR trend weakens (ADX < 20)
            if alligator_bear:
                exit_triggered = True
            if adx_val < 20:  # Trend weakening
                exit_triggered = True
            # Also exit if price falls below SMA200 in bull (bear confirmation)
            if not price_above_1d_sma200:
                exit_triggered = True
        
        if in_position and position_side < 0:
            # Exit short: alligator turns bullish OR trend weakens
            if alligator_bull:
                exit_triggered = True
            if adx_val < 20:
                exit_triggered = True
            # Exit if price rises above SMA200 in bear
            if price_above_1d_sma200:
                exit_triggered = True
        
        if exit_triggered:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                bars_in_trade = 0
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
            else:
                # Same direction - maintain position
                bars_in_trade += 1
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                bars_in_trade = 0
        
        signals[i] = desired_signal
    
    return signals