#!/usr/bin/env python3
"""
Experiment #021: 12h Williams Alligator + Volume Wake Up

HYPOTHESIS: The Williams Alligator (3 smoothed MAs) shows institutional 
"sleeping/waking" cycles. When the Alligator sleeps (lines converge), 
price compresses. When it wakes (lines separate), a move begins. Combined 
with volume confirmation on the wake-up bar, this catches major breakouts 
after consolidation. Works in both bull (buy wake-ups with bullish 1d trend) 
and bear (short rallies to Alligator in downtrend).

12h timeframe is slower than 4h, reducing trades from ~100 to ~50/year.
1d HMA adds trend bias to filter counter-trend entries.

TIMEFRAME: 12h primary
HTF: 1d for trend bias
TARGET: 75-150 total trades over 4 years (19-37/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_alligator_wakeup_1d_v1"
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

def calculate_williams_alligator(high, low, close):
    """
    Williams Alligator: 3 smoothed MAs
    Jaw (blue): SMMA(13) of median price
    Teeth (red): SMMA(8) of median price
    Lips (green): SMMA(5) of median price
    
    Returns: jaw, teeth, lips arrays
    """
    n = len(close)
    median = (high + low + close) / 3.0
    
    # SMMA using EWM with alpha = 1/period
    def smma(series, period):
        result = np.full(n, np.nan, dtype=np.float64)
        alpha = 1.0 / period
        running_sum = 0.0
        count = 0
        for i in range(n):
            if count < period:
                running_sum += series[i]
                count += 1
                if count == period:
                    result[i] = running_sum / period
            else:
                result[i] = result[i-1] + alpha * (series[i] - result[i-1])
        return result
    
    jaw = smma(median, 13)
    teeth = smma(median, 8)
    lips = smma(median, 5)
    
    return jaw, teeth, lips

def calculate_rsi(close, period=14):
    """RSI"""
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = (100 - (100 / (1 + rs))).values
    return rsi

def calculate_volatility_ratio(high, low, close, period=10):
    """Volatility Ratio for squeeze detection"""
    n = len(close)
    vr = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        hlt = high[i] - low[i]
        if hlt > 0:
            hmc = abs(close[i] - low[i - period] if i >= period else close[i])
            vr[i] = hlt / (hmc + 1e-10)
    
    return vr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d HMA for trend bias (bullish when price > HMA)
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate local 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Williams Alligator
    jaw, teeth, lips = calculate_williams_alligator(high, low, close)
    
    # RSI for momentum
    rsi_14 = calculate_rsi(close, period=14)
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Volatility ratio for squeeze detection
    vr = calculate_volatility_ratio(high, low, close, period=10)
    
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
    entry_bar = 0
    
    warmup = 100
    
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
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === ALLIGATOR STATE ===
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        
        # Alligator awake: lips > teeth > jaw (bullish) or lips < teeth < jaw (bearish)
        alligator_awake_bull = lips_val > teeth_val and teeth_val > jaw_val
        alligator_awake_bear = lips_val < teeth_val and teeth_val < jaw_val
        
        # Alligator sleeping: lines within 0.5 ATR of each other (consolidation)
        spread = abs(lips_val - jaw_val)
        alligator_sleeping = spread < 0.5 * atr_14[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5  # 50% above average
        
        # === TREND BIAS (1d HMA) ===
        price_above_1d_hma = close[i] > hma_1d_aligned[i]
        
        # === RSI FOR MOMENTUM ===
        rsi_val = rsi_14[i]
        
        # === ATR FOR STOPLOSS ===
        current_atr = atr_14[i]
        
        desired_signal = 0.0
        
        if not in_position:
            # === NEW LONG ENTRY ===
            # Price breaks above all lines + volume spike + bullish 1d trend
            if close[i] > lips_val and close[i] > teeth_val and close[i] > jaw_val:
                if vol_spike and price_above_1d_hma:
                    desired_signal = SIZE
            
            # === NEW SHORT ENTRY ===
            # Price breaks below all lines + volume spike + bearish 1d trend
            if close[i] < lips_val and close[i] < teeth_val and close[i] < jaw_val:
                if vol_spike and not price_above_1d_hma:
                    desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR trailing) ===
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
            # Long exit: price falls below alligator lines OR RSI oversold
            if close[i] < lips_val and close[i] < teeth_val:
                exit_triggered = True
            if rsi_val < 30:
                exit_triggered = True
        
        if in_position and position_side < 0:
            # Short exit: price rises above alligator lines OR RSI overbought
            if close[i] > lips_val and close[i] > teeth_val:
                exit_triggered = True
            if rsi_val > 70:
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