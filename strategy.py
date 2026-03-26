#!/usr/bin/env python3
"""
Experiment #023: 6h VWAP Anchor + Williams %R Regime + Volume Conf

HYPOTHESIS: 1d VWAP is a major institutional reference point that price 
reverts to after volatility expansions. Combined with Williams %R to detect 
oversold/overbought extremes (different from RSI), and volume confirmation, 
this captures mean reversion trades at key institutional levels.

Why 6h + VWAP: 
- 6h gives enough bars for Williams %R (14) to have valid readings
- 1d VWAP anchor provides strong institutional reference
- Works in both bull (long near VWAP in uptrend) and bear (short rallies to VWAP)

TARGET: 75-150 total trades over 4 years (19-37/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_vwap_williams_vol_1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_vwap(high, low, close, volume):
    """VWAP - Volume Weighted Average Price"""
    n = len(close)
    typical_price = (high + low + close) / 3.0
    cumulative_tp_vol = np.cumsum(typical_price * volume)
    cumulative_vol = np.cumsum(volume)
    vwap = np.zeros(n)
    for i in range(n):
        if cumulative_vol[i] > 0:
            vwap[i] = cumulative_tp_vol[i] / cumulative_vol[i]
    return vwap

def calculate_vwap_std(vwap, high, low, close, volume, period=20):
    """Standard deviation bands around VWAP"""
    n = len(close)
    typical_price = (high + low + close) / 3.0
    # Compute rolling std of price-distance from VWAP
    deviations = typical_price - vwap
    std_dev = pd.Series(deviations).rolling(window=period, min_periods=period).std().values
    return std_dev

def calculate_williams_r(high, low, close, period=14):
    """Williams %R - momentum oscillator"""
    n = len(close)
    williams_r = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        highest_high = np.max(high[i - period:i + 1])
        lowest_low = np.min(low[i - period:i + 1])
        
        if highest_high - lowest_low > 0:
            williams_r[i] = -100 * (highest_high - close[i]) / (highest_high - lowest_low)
    
    return williams_r

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
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # 1d VWAP for institutional anchor
    df_1d['vwap'] = calculate_vwap(
        df_1d['high'].values, 
        df_1d['low'].values, 
        df_1d['close'].values,
        df_1d['volume'].values
    )
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, df_1d['vwap'].values)
    
    # Calculate local 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Local VWAP and bands
    vwap_local = calculate_vwap(high, low, close, volume)
    vwap_std = calculate_vwap_std(vwap_local, high, low, close, volume, period=20)
    
    # Williams %R
    williams_r = calculate_williams_r(high, low, close, period=14)
    
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Deviation from VWAP for mean reversion signal
    vwap_deviation = (close - vwap_local) / (vwap_std + 1e-10)
    
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
    
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(vwap_local[i]) or np.isnan(vwap_std[i]) or vwap_std[i] <= 1e-10:
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
        
        if np.isnan(williams_r[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND BIAS (1d HMA) ===
        price_above_1d_hma = close[i] > hma_1d_aligned[i]
        
        # === 1d VWAP ANCHOR ===
        vwap_1d = vwap_1d_aligned[i] if not np.isnan(vwap_1d_aligned[i]) else close[i]
        price_vs_vwap_1d = close[i] / vwap_1d - 1.0  # percentage deviation
        
        # === LOCAL VWAP + BANDS ===
        vwap = vwap_local[i]
        band_1_std = vwap_std[i]
        upper_band = vwap + 1.5 * band_1_std
        lower_band = vwap - 1.5 * band_1_std
        
        # Price relative to local VWAP
        price_above_vwap_local = close[i] > vwap
        price_below_vwap_local = close[i] < vwap
        
        # === WILLIAMS %R REGIME ===
        wr_val = williams_r[i]
        # Oversold: below -80 (potential long)
        # Overbought: above -20 (potential short)
        is_oversold = wr_val < -80
        is_overbought = wr_val > -20
        is_neutral_wr = -80 <= wr_val <= -20
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.4
        
        # === MEAN REVERSION SIGNALS ===
        # Price far from VWAP (potential reversion)
        far_from_vwap = abs(vwap_deviation[i]) > 1.2
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === NEW LONG ENTRY ===
            # Conditions:
            # 1. Williams %R oversold (< -80)
            # 2. Price bouncing from below local VWAP OR price below lower band
            # 3. Bullish 1d trend (price > 1d HMA)
            # 4. Volume spike confirmation
            
            long_conditions = (
                is_oversold and
                (price_below_vwap_local or close[i] < lower_band) and
                price_above_1d_hma and
                vol_spike
            )
            
            if long_conditions:
                desired_signal = SIZE
            
            # === NEW SHORT ENTRY ===
            # Conditions:
            # 1. Williams %R overbought (> -20)
            # 2. Price rejecting from above local VWAP OR price above upper band
            # 3. Bearish 1d trend (price < 1d HMA)
            # 4. Volume spike confirmation
            
            short_conditions = (
                is_overbought and
                (price_above_vwap_local or close[i] > upper_band) and
                not price_above_1d_hma and
                vol_spike
            )
            
            if short_conditions:
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
        
        # === EXIT: Williams %R returns to neutral OR VWAP cross ===
        exit_triggered = False
        
        if in_position and position_side > 0:
            # Long exit: Williams %R returns to neutral OR price crosses above VWAP
            if is_neutral_wr and wr_val > -50:
                exit_triggered = True
            if close[i] > upper_band:
                exit_triggered = True
        
        if in_position and position_side < 0:
            # Short exit: Williams %R returns to neutral OR price crosses below VWAP
            if is_neutral_wr and wr_val < -50:
                exit_triggered = True
            if close[i] < lower_band:
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
        
        signals[i] = desired_signal
    
    return signals