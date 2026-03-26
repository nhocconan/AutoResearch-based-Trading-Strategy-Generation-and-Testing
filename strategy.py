#!/usr/bin/env python3
"""
Experiment #011: 6h Elder Ray + 1d Trend Regime + Volume Spike

HYPOTHESIS: Elder Ray Index (Bull/Bear Power) captures momentum exhaustion points
better than simple breakouts. Combined with 1d trend regime and volume confirmation,
this should work in both bull and bear markets by entering when the dominant side
shows strength AND the opposing side shows exhaustion.

WHY THIS SHOULD WORK IN BOTH BULL AND BEAR:
- Bull markets: Long when Bull Power > 0 (bulls control) + Bear Power rising (bear exhaustion)
- Bear markets: Short when Bear Power < 0 (bears control) + Bull Power falling (bull exhaustion)
- 1d HMA provides regime filter (only trade with higher TF trend)
- Volume spike confirms institutional participation at entry

TARGET: 75-200 total trades over 4 years (12-37/year for 6h).
Entry conditions: 3 confluence factors (Elder Ray + 1d trend + volume) to avoid overtrading.

KEY DESIGN:
1. Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
2. Entry: BP > 0 + Bear Power > Bear Power[5] (bulls strong, bears weakening)
3. 1d HMA(21) for trend bias (only long if price > 1d HMA)
4. Volume spike > 1.5x 20-avg for confirmation
5. ATR trailing stop (2.5x ATR)
6. Signal: 0.28 discrete (not 1.0!)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_elder_ray_1d_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def calculate_ema(close, span):
    """Exponential Moving Average with min_periods"""
    return pd.Series(close).ewm(span=span, min_periods=span, adjust=False).mean().values

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

def calculate_elder_ray(high, low, close, period=13):
    """
    Elder Ray Index
    Bull Power = High - EMA(13)
    Bear Power = Low - EMA(13)
    """
    ema_13 = calculate_ema(close, period)
    bull_power = high - ema_13
    bear_power = low - ema_13
    return bull_power, bear_power

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1d data for trend bias - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for trend bias
    close_1d = df_1d['close'].values
    hma_1d_raw = calculate_hma(close_1d, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    bull_power, bear_power = calculate_elder_ray(high, low, close, period=13)
    
    # Volume moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # EMA for additional confirmation
    ema_8 = calculate_ema(close, 8)
    ema_21 = calculate_ema(close, 21)
    
    signals = np.zeros(n)
    SIZE = 0.28  # Discrete position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup for all indicators
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(bull_power[i]) or np.isnan(bear_power[i]):
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
        
        if np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND BIAS (1d HMA) ===
        price_above_1d_hma = close[i] > hma_1d_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === ELDER RAY MOMENTUM ===
        bp = bull_power[i]
        bep = bear_power[i]
        
        # Bear Power trend (5-bar lookback for exhaustion)
        bep_5ago = bear_power[i-5] if i >= 5 else bep
        
        # Bull Power trend (5-bar lookback for exhaustion)
        bp_5ago = bull_power[i-5] if i >= 5 else bp
        
        # EMA cross confirmation
        ema_bullish = ema_8[i] > ema_21[i]
        ema_bearish = ema_8[i] < ema_21[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: Bulls in control (BP > 0) + Bears exhausting (BP rising) + trend up + volume
        if price_above_1d_hma:
            if bp > 0 and bp > bp_5ago:  # Bull power positive and strengthening
                if vol_spike or ema_bullish:  # Volume OR EMA confirmation
                    desired_signal = SIZE
        
        # SHORT: Bears in control (BEP < 0) + Bulls exhausting (BEP falling) + trend down + volume
        if not price_above_1d_hma:
            if bep < 0 and bep < bep_5ago:  # Bear power negative and strengthening
                if vol_spike or ema_bearish:  # Volume OR EMA confirmation
                    desired_signal = -SIZE
        
        # === STOPLOSS CHECK (ATR trailing) ===
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
        
        # === TAKE PROFIT (opposite Elder Ray signal) ===
        tp_triggered = False
        if in_position and position_side > 0:
            # Exit if Bull Power turns negative (momentum lost)
            if bp < 0:
                tp_triggered = True
        
        if in_position and position_side < 0:
            # Exit if Bear Power turns positive (momentum lost)
            if bep > 0:
                tp_triggered = True
        
        if tp_triggered:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
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
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = desired_signal
    
    return signals

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