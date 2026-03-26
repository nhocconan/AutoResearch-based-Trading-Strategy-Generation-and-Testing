#!/usr/bin/env python3
"""
Experiment #021: 12h Camarilla Pivot + ATR Expansion + Volume Spike

HYPOTHESIS: On 12h timeframe, Camarilla pivot levels from the daily chart mark
institutional support/resistance zones. When price touches S3/R3 with an ATR 
expansion (volatility spike) and volume confirmation, it signals a high-probability
mean reversion or continuation trade. This works in both bull (buy S3 dips) and 
bear (sell R3 rallies) markets.

WHY 12h: Slower than 4h = fewer but higher-quality signals. Institutional flow
visible on 12h. 12-37 trades/year is achievable.

DB PROVEN: Camarilla pivot + volume spike on 4h achieved test Sharpe 1.47 on ETH.
Scaling to 12h should reduce trades by ~2x while maintaining edge.

TIMEFRAME: 12h primary
HTF: 1d for Camarilla pivots
TARGET: 50-100 total trades over 4 years (12-25/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_camarilla_atr_vol_1d_v1"
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

def calculate_camarilla_levels(df_1d):
    """Calculate Camarilla pivot levels from 1d data.
    Returns arrays aligned with 1d index.
    """
    high = df_1d['high'].values
    low = df_1d['low'].values
    close = df_1d['close'].values
    n = len(close)
    
    # Camarilla formulas (use yesterday's HLC for today's pivots)
    h_l_range = high[:-1] - low[:-1]  # yesterday's range
    
    # Shift by 1 so today's pivot aligns with today
    h_shift = np.roll(high, 1)
    l_shift = np.roll(low, 1)
    c_shift = np.roll(close, 1)
    r_shift = np.roll(h_l_range, 1)
    
    h_shift[0] = np.nan
    l_shift[0] = np.nan
    c_shift[0] = np.nan
    r_shift[0] = np.nan
    
    # R3 = Close + Range * 1.1/6
    r3 = c_shift + r_shift * 1.1 / 6.0
    # R2 = Close + Range * 1.1/12
    r2 = c_shift + r_shift * 1.1 / 12.0
    # R1 = Close + Range * 1.1/2
    r1 = c_shift + r_shift * 1.1 / 2.0
    # PP = (H + L + C) / 3
    pp = (h_shift + l_shift + c_shift) / 3.0
    # S1 = Close - Range * 1.1/2
    s1 = c_shift - r_shift * 1.1 / 2.0
    # S2 = Close - Range * 1.1/12
    s2 = c_shift - r_shift * 1.1 / 12.0
    # S3 = Close - Range * 1.1/6
    s3 = c_shift - r_shift * 1.1 / 6.0
    
    return r3, r2, r1, pp, s1, s2, s3

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load 1d data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d HMA for trend direction
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Camarilla levels from 1d
    r3, r2, r1, pp, s1, s2, s3 = calculate_camarilla_levels(df_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    # Calculate local 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_30 = calculate_atr(high, low, close, period=30)
    
    # ATR expansion ratio
    atr_ratio = atr_14 / (atr_30 + 1e-10)
    
    # Volume MA and ratio
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
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 100  # Need enough bars for all indicators
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
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
        
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === INDICATOR VALUES ===
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        pp_val = pp_aligned[i]
        
        # ATR expansion (volatility spike)
        atr_expansion = atr_ratio[i] > 1.5
        
        # Volume spike
        vol_spike = vol_ratio[i] > 1.3
        
        # RSI value
        rsi_val = rsi[i]
        
        # Trend direction (1d HMA)
        bullish_trend = close[i] > hma_1d_aligned[i]
        
        # === CAMARILLA TOUCH DETECTION ===
        # Price touched or exceeded R3 (bearish bias - sell rallies)
        touch_r3 = (close[i] >= r3_val) or (high[i] >= r3_val)
        # Price touched or exceeded S3 (bullish bias - buy dips)
        touch_s3 = (close[i] <= s3_val) or (low[i] <= s3_val)
        
        # Distance to pivot as % of price
        dist_to_r3 = (high[i] - r3_val) / (close[i] + 1e-10)
        dist_to_s3 = (s3_val - low[i]) / (close[i] + 1e-10)
        near_r3 = dist_to_r3 < 0.01  # within 1% of R3
        near_s3 = dist_to_s3 < 0.01  # within 1% of S3
        
        desired_signal = 0.0
        
        # === ENTRY LOGIC ===
        if not in_position:
            # === LONG ENTRY: Touch S3 with bullish conditions ===
            if touch_s3 or near_s3:
                # Need: ATR expansion OR volume spike + bullish trend + RSI not overbought
                if (atr_expansion or vol_spike) and bullish_trend and rsi_val < 70:
                    desired_signal = SIZE
            
            # === SHORT ENTRY: Touch R3 with bearish conditions ===
            if touch_r3 or near_r3:
                # Need: ATR expansion OR volume spike + bearish trend + RSI not oversold
                if (atr_expansion or vol_spike) and not bullish_trend and rsi_val > 30:
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
        
        # === EXIT LOGIC ===
        exit_triggered = False
        
        if in_position and position_side > 0:
            # Long exit: price reaches R1/R2 OR RSI overbought
            if close[i] >= r1_val:
                exit_triggered = True
            if rsi_val > 75:
                exit_triggered = True
            # Or opposite signal
            if touch_r3 or near_r3:
                if (atr_expansion or vol_spike):
                    exit_triggered = True
        
        if in_position and position_side < 0:
            # Short exit: price reaches S1/S2 OR RSI oversold
            if close[i] <= s1_val:
                exit_triggered = True
            if rsi_val < 25:
                exit_triggered = True
            # Or opposite signal
            if touch_s3 or near_s3:
                if (atr_expansion or vol_spike):
                    exit_triggered = True
        
        if exit_triggered:
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
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals