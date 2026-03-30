#!/usr/bin/env python3
"""
Experiment #005: 12h Williams %R + KAMA Trend + Volume Confluence

HYPOTHESIS: Williams %R catches momentum reversals better than RSI because
it measures position within the range, not rate of change. Combined with
KAMA (adaptive smoothing) and volume confirmation, this should work in
bull, bear, and range markets.

WHY IT SHOULD WORK:
- Bull: %R oversold + KAMA up = strong reversal long
- Bear: %R overbought + KAMA down = continuation short
- Range: %R extremes + volume spike = mean reversion works

KEY DIFFERENCES FROM FAILED STRATEGIES:
- Williams %R (not RSI/CRSI) = different signal source
- KAMA (not EMA) = adaptive smoothing reduces lag
- Tighter entry: requires BOTH %R extreme AND volume spike
- ATR-based regime filter: scale down in high volatility

TARGET: 75-150 total trades over 4 years (19-37/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_williams_kama_vol_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_williams_r(high, low, close, period=14):
    """Williams %R - momentum oscillator measuring position within N-period range"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    wr = np.full(n, np.nan)
    for i in range(period - 1, n):
        if highest_high[i] != lowest_low[i]:
            wr[i] = -100 * (highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i])
    
    return wr

def calculate_kama(close, period=21, fast_ema=2, slow_ema=30):
    """
    Kaufman Adaptive Moving Average
    Adapts to market volatility - fast in trending, slow in ranging
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    # Calculate Efficiency Ratio (ER)
    er = np.full(n, np.nan)
    direction = np.abs(close[period:] - close[:-period])
    volatility = np.abs(close[period:] - close[:-1])
    vol_sum = np.zeros(n)
    
    for i in range(period, n):
        vol_sum[i] = np.sum(volatility[i - period + 1:i + 1])
        if vol_sum[i] > 0:
            er[i] = direction[i] / vol_sum[i]
    
    # Calculate smoothing constants
    fast_const = 2 / (fast_ema + 1)
    slow_const = 2 / (slow_ema + 1)
    smoothing = er * (fast_const - slow_const) + slow_const
    smoothing_squared = smoothing * smoothing
    
    kama = np.full(n, np.nan)
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        if not np.isnan(smoothing_squared[i]):
            kama[i] = kama[i - 1] + smoothing_squared[i] * (close[i] - kama[i - 1])
    
    return kama

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

def calculate_volume_ratio(volume, period=20):
    """Volume relative to moving average"""
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return volume / np.where(vol_ma > 0, vol_ma, 1)

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d KAMA for trend direction
    kama_21_1d = calculate_kama(df_1d['close'].values, period=21)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama_21_1d)
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    williams_r = calculate_williams_r(high, low, close, period=14)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
    # ATR regime: high volatility = reduce position
    atr_ma = pd.Series(atr_14).rolling(window=30, min_periods=30).mean().values
    atr_regime = atr_14 / np.where(atr_ma > 0, atr_ma, 1)  # >1.5 = high vol
    
    signals = np.zeros(n)
    SIZE = 0.28  # Base position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 250  # Need 200 for any rolling + 14 for Williams R + 20 for volume MA
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(williams_r[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(kama_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME CHECKS ===
        wr_value = williams_r[i]
        vol_spike = vol_ratio[i] > 1.5
        atr_scale = 1.0 if atr_regime[i] < 1.3 else 0.7  # Reduce in high vol
        
        # === 1d TREND: KAMA direction ===
        kama_trend_up = close[i] > kama_aligned[i]
        kama_trend_down = close[i] < kama_aligned[i]
        
        # === WILLIAMS %R SIGNALS ===
        # Oversold: %R < -80 (reversal long potential)
        # Overbought: %R > -20 (reversal short potential)
        # Neutral: -80 to -20
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: %R oversold + KAMA up + volume spike ===
            if wr_value < -80 and kama_trend_up and vol_spike:
                desired_signal = SIZE * atr_scale
            
            # === LONG: Extreme oversold + strong trend (smaller size, no vol) ===
            elif wr_value < -90 and kama_trend_up:
                desired_signal = SIZE * 0.6 * atr_scale
            
            # === SHORT: %R overbought + KAMA down + volume spike ===
            if wr_value > -20 and kama_trend_down and vol_spike:
                desired_signal = -SIZE * atr_scale
            
            # === SHORT: Extreme overbought + strong trend (smaller size, no vol) ===
            elif wr_value > -10 and kama_trend_down:
                desired_signal = -SIZE * 0.6 * atr_scale
        
        # === STOPLOSS (2.5 ATR) ===
        if in_position:
            if position_side > 0:
                # Update highest high since entry
                if i == entry_bar or high[i] > highest_since_entry:
                    highest_since_entry = high[i]
                
                # Trailing stop
                stop_price = highest_since_entry - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Exit if KAMA trend flips
                if kama_trend_down:
                    desired_signal = 0.0
                
                # Exit if %R reaches overbought without continuation
                if wr_value > -20 and vol_ratio[i] < 1.0:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Update lowest low since entry
                if i == entry_bar or low[i] < lowest_since_entry:
                    lowest_since_entry = low[i]
                
                # Trailing stop
                stop_price = lowest_since_entry + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Exit if KAMA trend flips
                if kama_trend_up:
                    desired_signal = 0.0
                
                # Exit if %R reaches oversold without continuation
                if wr_value < -80 and vol_ratio[i] < 1.0:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 4 bars to avoid fee churn ===
        if in_position and (i - entry_bar) < 4:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals