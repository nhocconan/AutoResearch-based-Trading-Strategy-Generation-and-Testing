#!/usr/bin/env python3
"""
Experiment #011: 6h Williams Alligator + ADX + 1d Trend

HYPOTHESIS: Williams Alligator (Jaw=SMA13, Teeth=SMA8, Lips=SMA5) provides
natural multi-component trend structure. When Lips crosses above Teeth
(both above Jaw) with ADX>20 confirming trend strength, and 1d SMA confirming
direction — this marks institutional trend entries. The shifted EMAs (8/5/3 bars)
create built-in momentum confirmation without additional oscillators.

TIMEFRAME: 6h primary
HTF: 1d for trend bias, 1w for regime
TARGET: 75-150 total trades over 4 years (19-37/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_alligator_adx_1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_alligator(high, low, close, jaw_period=13, teeth_period=8, lips_period=5):
    """
    Williams Alligator indicator.
    Jaw = SMA(13) of median price, shifted 8 bars
    Teeth = SMA(8) of median price, shifted 5 bars
    Lips = SMA(5) of median price, shifted 3 bars
    """
    n = len(close)
    median = (high + low + close) / 3.0
    
    # Calculate raw SMAs
    jaw_raw = pd.Series(median).rolling(window=jaw_period, min_periods=jaw_period).mean().values
    teeth_raw = pd.Series(median).rolling(window=teeth_period, min_periods=teeth_period).mean().values
    lips_raw = pd.Series(median).rolling(window=lips_period, min_periods=lips_period).mean().values
    
    # Apply Williams shift (shift forward = delay in time series)
    jaw = np.roll(jaw_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    
    # Set NaN for first elements
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    return jaw, teeth, lips

def calculate_adx(high, low, close, period=14):
    """ADX with DMI"""
    n = len(close)
    if n < period * 2:
        return np.full(n, np.nan)
    
    # True Range
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Directional Movement
    up_move = high[i] - high[i-1]
    down_move = low[i-1] - low[i]
    
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        up = high[i] - high[i-1]
        down = low[i-1] - low[i]
        plus_dm[i] = up if (up > down and up > 0) else 0
        minus_dm[i] = down if (down > up and down > 0) else 0
    
    # Smooth with EMA
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_di = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # DX
    di_sum = plus_di + minus_di + 1e-10
    dx = 100 * np.abs(plus_di - minus_di) / di_sum
    
    # ADX = smoothed DX
    adx = pd.Series(dx).ewm(span=period, min_periods=period * 2, adjust=False).mean().values
    
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
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # 1d SMA for trend bias
    sma_1d_raw = pd.Series(df_1d['close'].values).rolling(window=50, min_periods=50).mean().values
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d_raw)
    
    # 1w SMA for regime
    sma_1w_raw = pd.Series(df_1w['close'].values).rolling(window=21, min_periods=21).mean().values
    sma_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_1w_raw)
    
    # Calculate local 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    adx = calculate_adx(high, low, close, period=14)
    
    # Williams Alligator
    jaw, teeth, lips = calculate_alligator(high, low, close, jaw_period=13, teeth_period=8, lips_period=5)
    
    # Volume MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # RSI for momentum
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, min_periods=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = (100 - (100 / (1 + rs))).values
    
    signals = np.zeros(n)
    SIZE = 0.30
    SIZE_HALF = 0.15
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
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
        
        if np.isnan(sma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === ALLIGATOR STATE ===
        alligator_awake_long = lips[i] > teeth[i] and teeth[i] > jaw[i]  # Lines aligned up
        alligator_awake_short = lips[i] < teeth[i] and teeth[i] < jaw[i]  # Lines aligned down
        alligator_closed = abs(lips[i] - jaw[i]) < 0.1 * jaw[i]  # Alligator sleeping
        
        # Price vs Alligator
        price_above_alligator = close[i] > lips[i] and close[i] > teeth[i] and close[i] > jaw[i]
        price_below_alligator = close[i] < lips[i] and close[i] < teeth[i] and close[i] < jaw[i]
        
        # === 1d TREND BIAS ===
        price_above_1d_sma = close[i] > sma_1d_aligned[i]
        
        # === 1w REGIME ===
        price_above_1w_sma = close[i] > sma_1w_aligned[i] if not np.isnan(sma_1w_aligned[i]) else True
        
        # === ADX TREND STRENGTH ===
        adx_val = adx[i] if not np.isnan(adx[i]) else 0
        strong_trend = adx_val > 20
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.2
        
        # === RSI MOMENTUM ===
        rsi_val = rsi[i]
        
        # === LIPS CROSS TEETH (momentum shift) ===
        # Check if lips crosses above teeth (long momentum)
        lips_cross_above_teeth = (lips[i] > teeth[i] and lips[i-1] <= teeth[i-1]) if i > 1 else False
        # Check if lips crosses below teeth (short momentum)
        lips_cross_below_teeth = (lips[i] < teeth[i] and lips[i-1] >= teeth[i-1]) if i > 1 else False
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === NEW LONG ENTRY ===
            # Momentum shift up + alligator aligned + trend aligned
            if lips_cross_above_teeth and alligator_awake_long and price_above_1d_sma:
                # Add trend strength filter
                if strong_trend:
                    desired_signal = SIZE
            
            # Alternative: Price breaks above all lines with volume
            if price_above_alligator and price_above_1d_sma and vol_spike and strong_trend:
                if alligator_awake_long:
                    desired_signal = SIZE
            
            # === NEW SHORT ENTRY ===
            # Momentum shift down + alligator aligned down + trend aligned down
            if lips_cross_below_teeth and alligator_awake_short and not price_above_1d_sma:
                if strong_trend:
                    desired_signal = -SIZE
            
            # Alternative: Price breaks below all lines with volume
            if price_below_alligator and not price_above_1d_sma and vol_spike and strong_trend:
                if alligator_awake_short:
                    desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR) ===
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
        
        # === TAKE PROFIT at 2R ===
        tp_triggered = False
        if in_position and position_side > 0:
            profit_pct = (high[i] - entry_price) / entry_price
            if profit_pct > 0.04:  # 4% = 2R approx (ATR is typically ~2%)
                tp_triggered = True
        
        if in_position and position_side < 0:
            profit_pct = (entry_price - low[i]) / entry_price
            if profit_pct > 0.04:
                tp_triggered = True
        
        if tp_triggered:
            # Reduce to half position
            if in_position:
                desired_signal = SIZE_HALF * position_side
        
        # === EXIT CONDITIONS ===
        exit_triggered = False
        
        if in_position and position_side > 0:
            # Long exit: price crosses below alligator OR RSI oversold
            if price_below_alligator and lips[i] < teeth[i]:
                exit_triggered = True
            if rsi_val < 35:
                exit_triggered = True
            # Alligator closing (reversal signal)
            if alligator_awake_short:
                exit_triggered = True
        
        if in_position and position_side < 0:
            # Short exit: price crosses above alligator OR RSI overbought
            if price_above_alligator and lips[i] > teeth[i]:
                exit_triggered = True
            if rsi_val > 65:
                exit_triggered = True
            # Alligator closing (reversal signal)
            if alligator_awake_long:
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
                entry_bar = i
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
            else:
                # Same direction - maintain position
                pass
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                entry_bar = 0
        
        signals[i] = desired_signal
    
    return signals