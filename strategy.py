#!/usr/bin/env python3
"""
Experiment #022: 12h Williams %R Extreme + KAMA + ATR Regime

HYPOTHESIS: Williams %R at extreme oversold (<-85) or overbought (>-15) 
captures reversal points. Combined with 1d KAMA trend direction and 1d ATR
regime filter (high vol = better for reversals) + volume confirmation, 
this catches major turning points while filtering noise.

WHY 12h: 3x slower than 4h = fewer but higher-quality trades.
WHY KAMA: Adaptive to volatility, smoother than EMA.
WHY ATR REGIME: High volatility regime favors mean reversion.

TARGET: 75-150 total over 4 years (19-37/year). HARD MAX: 200.
Signal size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_willr_kama_atr_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_williams_r(high, low, close, period=14):
    """Williams %R"""
    n = len(close)
    result = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        if highest_high != lowest_low:
            result[i] = -100 * (highest_high - close[i]) / (highest_high - lowest_low)
        else:
            result[i] = -50
    
    return result

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

def calculate_kama(prices, period=21):
    """Kaufman Adaptive Moving Average"""
    n = len(prices)
    kama = np.full(n, np.nan)
    
    # Need at least period+1 for first AMA
    if n < period + 1:
        return kama
    
    # Use high for trend detection
    close = prices if isinstance(prices, np.ndarray) else prices.values
    
    # Efficiency Ratio (ER)
    change = np.abs(close[period:] - close[:-period])
    volatility = np.zeros(n - period)
    for i in range(n - period):
        volatility[i] = np.sum(np.abs(close[i+1:i+period+1] - close[i:i+period]))
    
    er = np.zeros(n)
    er[period:] = change / np.maximum(volatility, 1e-10)
    
    # Fast and slow SC
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # First KAMA value is SMA
    kama[period] = np.mean(close[:period + 1])
    
    # KAMA calculation
    for i in range(period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    SIZE = 0.25
    
    # === 1d indicators (HTF) ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d ATR for regime detection
    atr_1d = calculate_atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=14)
    atr_ma_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_ratio_1d = atr_1d / np.maximum(atr_ma_1d, 1e-10)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    # 1d KAMA for trend
    kama_1d_raw = calculate_kama(df_1d['close'].values, period=21)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    # === Primary TF (12h) indicators ===
    willr_14 = calculate_williams_r(high, low, close, period=14)
    
    # ATR for stoploss
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.maximum(vol_ma, 1e-10)
    
    # === Signal generation ===
    signals = np.zeros(n)
    
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
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(atr_ratio_aligned[i]) or np.isnan(kama_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Current values
        willr = willr_14[i]
        price_above_kama = close[i] > kama_aligned[i]
        vol_spike = vol_ratio[i] > 1.8
        high_vol_regime = atr_ratio_aligned[i] > 1.3
        
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Williams %R extreme oversold + KAMA uptrend + vol spike + high vol ===
            if willr < -85 and price_above_kama and vol_spike and high_vol_regime:
                desired_signal = SIZE
            
            # === SHORT: Williams %R extreme overbought + KAMA downtrend + vol spike + high vol ===
            if willr > -15 and not price_above_kama and vol_spike and high_vol_regime:
                desired_signal = -SIZE
        
        # === STOPLOSS (2.0 ATR trailing) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                desired_signal = 0.0
        
        # === HOLD PERIOD (minimum 2 bars = 1 day to avoid churn) ===
        bars_held = i - entry_bar
        
        if in_position and bars_held >= 2:
            # Take profit: Williams %R mean reversion to -50
            if position_side > 0 and willr > -50:
                desired_signal = 0.0
            if position_side < 0 and willr < -50:
                desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                entry_bar = i
                if position_side > 0:
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                stop_price = 0.0
        
        signals[i] = desired_signal
    
    return signals