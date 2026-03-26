#!/usr/bin/env python3
"""
Experiment #004: 1d KAMA + RSI + Volume Confirmation + 1w Trend

HYPOTHESIS: KAMA (adaptive moving average) filters noise better than SMA/EMA.
Combined with RSI extremes for momentum reversal signals, volume confirmation
for institutional involvement, and 1w HMA for multi-year trend direction.

WHY THIS SHOULD WORK IN BOTH BULL AND BEAR:
- KAMA adapts to volatility - fast in trends, slow in chop
- RSI extremes (oversold<30 / overbought>70) catch reversals in any direction
- Volume confirms institutional interest at key levels
- 1w trend filter prevents fighting major trends

TARGET: 50-100 total trades over 4 years (proven pattern from DB).
DB reference: mtf_1d_kama_rsi_chop_regime_1w_v1 (test Sharpe=1.310, 74 trades)

KEY DESIGN (keep simple - fewer conditions = fewer trades = less fee drag):
1. KAMA trend direction (bullish when KAMA rising, bearish when falling)
2. RSI extreme (oversold for longs, overbought for shorts)
3. Volume confirmation (>1.3x 20-day average)
4. 1w HMA for trend bias
5. ATR-based stoploss and take profit
6. Signal: 0.25-0.30 (discrete)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_rsi_vol_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_kama(close, period=21, fast_ema=2, slow_ema=30):
    """
    Kaufman Adaptive Moving Average
    Uses EMA efficiency ratio to adapt between fast and slow EMAs
    """
    n = len(close)
    if n < slow_ema + period:
        return np.full(n, np.nan)
    
    # Calculate price change (absolute)
    change = np.abs(close[period:] - close[:-period])
    
    # Calculate volatility (sum of absolute changes)
    volatility = np.zeros(n)
    for i in range(period, n):
        volatility[i] = np.sum(np.abs(np.diff(close[max(0, i-period):i+1])))
    
    # Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        if volatility[i] > 1e-10:
            er[i] = change[i - period] / volatility[i]
    
    # Smoothing constant
    fast_con = 2.0 / (fast_ema + 1)
    slow_con = 2.0 / (slow_ema + 1)
    fastest_con = 2.0 / (2 + 1)
    slowest_con = 2.0 / (30 + 1)
    
    sc = np.zeros(n)
    for i in range(n):
        if er[i] > 0:
            sc[i] = (er[i] * (fast_con - slow_con) + slow_con) ** 2
    
    # KAMA calculation
    kama = np.full(n, np.nan, dtype=np.float64)
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        if not np.isnan(sc[i]) and sc[i] > 0:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1] if not np.isnan(kama[i-1]) else close[i]
    
    return kama

def calculate_rsi(close, period=14):
    """RSI - Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    
    # First average
    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])
    
    # Smoothed averages
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i-1]) / period
    
    rsi = np.full(n, 50.0, dtype=np.float64)
    for i in range(period, n):
        if avg_loss[i] > 1e-10:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
    return rsi

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

def calculate_volume_ma(volume, period=20):
    """Simple volume moving average"""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === LOAD 1w DATA FOR TREND ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA for trend
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 1w SMA for additional confirmation
    sma_1w = pd.Series(df_1w['close'].values).rolling(window=21, min_periods=21).mean().values
    sma_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_1w)
    
    # === CALCULATE 1d INDICATORS ===
    kama_21 = calculate_kama(close, period=21)
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    vol_ma = calculate_volume_ma(volume, period=20)
    
    # Volume ratio
    vol_ratio = np.zeros(n)
    for i in range(n):
        if vol_ma[i] > 0 and not np.isnan(vol_ma[i]):
            vol_ratio[i] = volume[i] / vol_ma[i]
        else:
            vol_ratio[i] = 1.0
    
    # KAMA slope (trend direction)
    kama_slope = np.zeros(n)
    for i in range(5, n):
        if not np.isnan(kama_21[i]) and not np.isnan(kama_21[i-5]):
            kama_slope[i] = (kama_21[i] - kama_21[i-5]) / kama_21[i-5]
    
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    stop_price = 0.0
    
    warmup = 60
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(kama_21[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND BIAS (1w HMA) ===
        weekly_bullish = False
        weekly_bearish = False
        
        if not np.isnan(hma_1w_aligned[i]) and close[i] > hma_1w_aligned[i]:
            weekly_bullish = True
        if not np.isnan(hma_1w_aligned[i]) and close[i] < hma_1w_aligned[i]:
            weekly_bearish = True
        
        # === KAMA TREND ===
        kama_rising = kama_slope[i] > 0.0005
        kama_falling = kama_slope[i] < -0.0005
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        rsi_neutral_high = rsi_14[i] > 55  # For bearish entries
        rsi_neutral_low = rsi_14[i] < 45   # For bullish entries
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.3
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG ENTRY: KAMA rising + RSI oversold + weekly bullish + volume
        if weekly_bullish and kama_rising:
            if rsi_neutral_low and vol_spike:
                desired_signal = SIZE
            elif rsi_oversold:
                desired_signal = SIZE
        
        # SHORT ENTRY: KAMA falling + RSI overbought + weekly bearish + volume
        if weekly_bearish and kama_falling:
            if rsi_neutral_high and vol_spike:
                desired_signal = -SIZE
            elif rsi_overbought:
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK ===
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
        
        # === TAKE PROFIT at 3R ===
        tp_triggered = False
        if in_position and position_side > 0:
            profit = close[i] - entry_price
            if profit >= 3.0 * entry_atr:
                tp_triggered = True
        
        if in_position and position_side < 0:
            profit = entry_price - close[i]
            if profit >= 3.0 * entry_atr:
                tp_triggered = True
        
        if tp_triggered:
            desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or reversal
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
                # Exit if no longer meets entry criteria
                if position_side > 0:
                    # Exit long if KAMA turns bearish or RSI overbought
                    if kama_falling or rsi_14[i] > 75:
                        in_position = False
                        position_side = 0
                else:
                    # Exit short if KAMA turns bullish or RSI oversold
                    if kama_rising or rsi_14[i] < 25:
                        in_position = False
                        position_side = 0
        
        signals[i] = desired_signal if desired_signal != 0.0 else (SIZE * position_side if in_position else 0.0)
    
    return signals