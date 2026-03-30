#!/usr/bin/env python3
"""
Experiment #022: 6h RSI Extreme + ATR Volatility + 1d Trend

HYPOTHESIS: RSI(14) extremes (<30 long, >70 short) capture mean-reversion 
signals at exhaustion points. Combined with 1d SMA200 for trend direction and
ATR volatility filter, this captures high-probability reversals without 
overtrading. 6h timeframe balances opportunity vs fee drag.

WHY IT WORKS IN BOTH MARKETS:
- Bull: RSI < 30 = temporary dip, price bounces to fair value
- Bear: RSI > 70 = rallies to exhaustion, short continuation
- Range: RSI extremes work well in sideways markets
- ATR filter avoids "falling knife" entries in collapsing volatility

KEY DIFFERENCE FROM FAILED ATTEMPTS: Simple RSI extremes + strong trend filter
(1d SMA200) + volatility guard (ATR rising). Not stacking multiple weak signals.

TARGET: 75-150 total trades over 4 years (19-37/year). HARD MAX: 200.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_rsi_extreme_atr_vol_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

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

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA200 for trend direction
    sma_1d = pd.Series(df_1d['close'].values).rolling(window=200, min_periods=200).mean().values
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d)
    
    # Local 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_ma = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values
    rsi_14 = calculate_rsi(close, period=14)
    
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
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
    entry_bar = 0
    
    warmup = 250  # Need 200 for SMA200 + buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
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
        
        # === ATR VOLATILITY FILTER ===
        # Only enter when ATR is rising (above its MA) - confirms momentum
        atr_rising = atr_14[i] > atr_ma[i] if not np.isnan(atr_ma[i]) else True
        atr_ratio = atr_14[i] / atr_ma[i] if not np.isnan(atr_ma[i]) and atr_ma[i] > 0 else 1.5
        
        # === TREND DIRECTION (1d SMA200) ===
        price_above_1d_sma = close[i] > sma_1d_aligned[i]
        
        # === RSI SIGNALS ===
        rsi = rsi_14[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: RSI oversold + above trend + ATR rising + vol spike ===
            # RSI < 35 = oversold (slightly above 30 to catch more reversals)
            # ATR ratio > 1.1 = volatility expanding
            if rsi < 35 and price_above_1d_sma and atr_ratio > 1.1:
                if vol_spike:
                    desired_signal = SIZE
            
            # === SHORT: RSI overbought + below trend + ATR rising + vol spike ===
            # RSI > 65 = overbought (slightly below 70 to catch more reversals)
            if rsi > 65 and not price_above_1d_sma and atr_ratio > 1.1:
                if vol_spike:
                    desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR trailing) ===
        if in_position:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
                trailing_stop = highest_since_entry - 2.5 * entry_atr
                stop_price = max(stop_price, trailing_stop)
                if low[i] < stop_price:
                    desired_signal = 0.0
            
            if position_side < 0:
                lowest_since_entry = min(lowest_since_entry, low[i])
                trailing_stop = lowest_since_entry + 2.5 * entry_atr
                stop_price = min(stop_price, trailing_stop)
                if high[i] > stop_price:
                    desired_signal = 0.0
        
        # === ATR-BASED EXIT (profit taking) ===
        bars_held = i - entry_bar
        
        if in_position:
            # Exit if ATR contracts (volatility crush - take profit)
            if bars_held >= 4 and atr_ratio < 0.9:
                desired_signal = 0.0
            
            # Exit if RSI reverts to neutral
            if position_side > 0 and rsi > 55:
                desired_signal = 0.0
            if position_side < 0 and rsi < 45:
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