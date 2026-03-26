#!/usr/bin/env python3
"""
Experiment #021: 12h Williams %R Mean Reversion + ATR Regime

HYPOTHESIS: Williams %R measures momentum extremes - oversold (<-80) often 
bounces and overbought (>-20) often reverses. Combined with ATR regime filter 
(volume/volatility confirmation) and simple 1d SMA trend bias, this captures 
mean-reversion opportunities in both bull and bear markets.

- Long: %R < -80 (oversold) + ATR regime (high vol) + price > SMA200
- Short: %R > -20 (overbought) + ATR regime + price < SMA200
- Exit: %R crosses center OR opposite signal OR 2*ATR stoploss

TIMEFRAME: 12h primary
HTF: 1d for SMA trend bias
TARGET: 75-150 total trades over 4 years (19-37/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_williams_r_atr_regime_v1"
timeframe = "12h"
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

def calculate_williams_r(high, low, close, period=14):
    """Williams %R - momentum oscillator"""
    n = len(close)
    willr = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        period_high = np.max(high[i - period + 1:i + 1])
        period_low = np.min(low[i - period + 1:i + 1])
        if period_high != period_low:
            willr[i] = -100 * (period_high - close[i]) / (period_high - period_low)
    
    return willr

def calculate_sma(data, period):
    """Simple Moving Average"""
    return pd.Series(data).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA for trend bias
    sma_200_1d = calculate_sma(df_1d['close'].values, 200)
    sma_200_aligned = align_htf_to_ltf(prices, df_1d, sma_200_1d)
    
    # Calculate local 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    williams_r = calculate_williams_r(high, low, close, period=14)
    
    # ATR regime: compare recent ATR to longer-term ATR
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_ratio = atr_7 / np.where(atr_14 > 0, atr_14, 1)
    
    # Volume MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # RSI for additional confirmation
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
    
    warmup = 250
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
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
        
        if np.isnan(sma_200_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # Current values
        wr = williams_r[i]
        rsi_val = rsi[i]
        vol_r = vol_ratio[i]
        atr_r = atr_ratio[i]
        
        # Trend bias from 1d SMA
        price_above_200sma = close[i] > sma_200_aligned[i]
        price_below_200sma = close[i] < sma_200_aligned[i]
        
        # ATR regime: high volatility (>1.2x normal) = good for mean reversion
        high_vol_regime = atr_r > 1.2
        
        # Volume confirmation
        vol_confirm = vol_r > 1.2
        
        # === Williams %R thresholds ===
        # Oversold: < -80 (potential long bounce)
        # Overbought: > -20 (potential short reversal)
        wr_oversold = wr < -80
        wr_overbought = wr > -20
        
        # %R crossing up from oversold = bullish reversal
        wr_crossed_up = False
        if i > 0 and not np.isnan(williams_r[i-1]):
            if williams_r[i-1] < -80 and wr >= -80:
                wr_crossed_up = True
        
        # %R crossing down from overbought = bearish reversal
        wr_crossed_down = False
        if i > 0 and not np.isnan(williams_r[i-1]):
            if williams_r[i-1] > -20 and wr <= -20:
                wr_crossed_down = True
        
        desired_signal = 0.0
        
        if not in_position:
            # === NEW LONG ENTRY ===
            # Williams %R oversold and starting to bounce + high vol regime + uptrend
            if wr_oversold and price_above_200sma and high_vol_regime and vol_confirm:
                desired_signal = SIZE
            
            # Alternative: %R crossing up from oversold (more aggressive)
            if wr_crossed_up and price_above_200sma:
                desired_signal = SIZE
            
            # === NEW SHORT ENTRY ===
            # Williams %R overbought and starting to drop + high vol regime + downtrend
            if wr_overbought and price_below_200sma and high_vol_regime and vol_confirm:
                desired_signal = -SIZE
            
            # Alternative: %R crossing down from overbought (more aggressive)
            if wr_crossed_down and price_below_200sma:
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
        
        # === EXIT: %R returns to neutral zone OR opposite signal ===
        exit_triggered = False
        
        if in_position and position_side > 0:
            # Long exit: %R back above -20 (overbought) OR RSI overbought
            if wr > -20:
                exit_triggered = True
            if rsi_val > 70:
                exit_triggered = True
        
        if in_position and position_side < 0:
            # Short exit: %R back below -80 (oversold) OR RSI oversold
            if wr < -80:
                exit_triggered = True
            if rsi_val < 30:
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
                # Same direction - maintain position (no churn)
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