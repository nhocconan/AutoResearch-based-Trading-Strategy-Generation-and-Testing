#!/usr/bin/env python3
"""
Experiment #013: 4h RSI Extreme + Volume + 1d Trend Filter

HYPOTHESIS: RSI extremes (below 30 / above 70) capture mean-reversion setups
at key support/resistance. Combined with volume confirmation and 1d SMA(50)
trend alignment, this works in BOTH bull markets (long RSI oversold bounces)
and bear markets (short RSI overbought in downtrend). 4h timeframe provides
enough signal diversity while keeping trades manageable.

WHY THIS SHOULD WORK:
- RSI 30/70 extremes are well-defined, statistically robust entry points
- Volume confirmation filters false signals
- 1d SMA(50) provides directional bias without being too slow
- Works in bull (long bounces off support) AND bear (short rallies to SMA)
- 4h = 1095 bars/year, RSI 30/70 should trigger 75-200 times/year

TIMEFRAME: 4h primary
HTF: 1d for trend SMA
TARGET: 75-200 total trades over 4 years (19-50/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_rsi_extreme_vol_1d_v1"
timeframe = "4h"
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA(50) for trend - simple, reliable
    sma_1d_raw = pd.Series(df_1d['close'].values).rolling(window=50, min_periods=50).mean().values
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d_raw)
    
    # Local 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, min_periods=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = (100 - (100 / (1 + rs))).values
    
    # Volume MA(20)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Bollinger Bands for additional confirmation
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + 2.0 * std_20
    bb_lower = sma_20 - 2.0 * std_20
    
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
    
    warmup = 60
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi[i]):
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
        
        rsi_val = rsi[i]
        vol_val = vol_ratio[i]
        
        # === TREND DIRECTION (1d SMA) ===
        price_above_1d_sma = close[i] > sma_1d_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_confirm = vol_val > 1.2
        
        # === BB POSITION FOR MEAN REVERSION TARGET ===
        bb_pos = (close[i] - bb_lower[i]) / (bb_upper[i] - bb_lower[i] + 1e-10) if not np.isnan(bb_lower[i]) else 0.5
        
        desired_signal = 0.0
        
        # === NEW ENTRY CONDITIONS ===
        if not in_position:
            # === LONG: RSI oversold + volume spike + price above 1d SMA ===
            # RSI below 30 = oversold, price bouncing from lower BB area
            if rsi_val < 35 and rsi_val > 10:  # Oversold but not dead
                if vol_confirm and price_above_1d_sma:
                    desired_signal = SIZE
            
            # === SHORT: RSI overbought + volume spike + price below 1d SMA ===
            # RSI above 70 = overbought, price rejected from upper BB area
            if rsi_val > 65 and rsi_val < 95:  # Overbought but not exhausted
                if vol_confirm and not price_above_1d_sma:
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
        
        # === EXIT CONDITIONS ===
        exit_triggered = False
        
        if in_position and position_side > 0:
            # Long exit: RSI normalizes (above 50) or price reaches BB upper
            if rsi_val > 55:
                exit_triggered = True
            # Take profit: price moved 2.5 ATR in our favor
            if close[i] > entry_price + 2.5 * entry_atr:
                exit_triggered = True
        
        if in_position and position_side < 0:
            # Short exit: RSI normalizes (below 50) or price reaches BB lower
            if rsi_val < 45:
                exit_triggered = True
            # Take profit: price moved 2.5 ATR in our favor
            if close[i] < entry_price - 2.5 * entry_atr:
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