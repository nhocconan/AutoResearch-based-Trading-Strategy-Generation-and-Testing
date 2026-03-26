#!/usr/bin/env python3
"""
Experiment #024: 4h Williams %R Reversal + 1d Trend + ATR Stoploss

HYPOTHESIS: Williams %R(14) at extremes (-80/+20) marks momentum exhaustion
and potential reversal points. Combined with 1d SMA200 trend filter, this
captures mean-reversion trades WITH trend alignment.

WHY IT WORKS IN BOTH BULL AND BEAR:
- Bull market: Price repeatedly drops to -80 in uptrends = excellent long entries
- Bear market: -80 fails to recover = avoids bad longs, short rallies at +20
- Williams %R adapts naturally: doesn't need explicit regime detection

TIMEFRAME: 4h primary, 1d HTF for trend
TARGET: 75-200 total trades over 4 years
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_williams_r_sma200_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_williams_r(high, low, close, period=14):
    """Williams %R - momentum oscillator for reversals"""
    n = len(close)
    williams = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        if highest_high != lowest_low:
            williams[i] = -100 * (highest_high - close[i]) / (highest_high - lowest_low)
        else:
            williams[i] = -50  # neutral when range is zero
    
    return williams

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
    
    # 1d SMA200 for trend filter (align to 4h)
    sma_200_1d = df_1d['close'].rolling(window=200, min_periods=200).mean().values
    sma_200_aligned = align_htf_to_ltf(prices, df_1d, sma_200_1d)
    
    # Calculate local 4h indicators
    williams_r = calculate_williams_r(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Previous Williams %R for cross detection
    williams_r_prev = np.roll(williams_r, 1)
    williams_r_prev[0] = np.nan
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
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
    
    warmup = 250  # Need enough for SMA200 on 1d aligned to 4h
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(williams_r[i]) or np.isnan(williams_r_prev[i]):
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
        
        wr = williams_r[i]
        wr_prev = williams_r_prev[i]
        
        # === TREND FILTER (1d SMA200) ===
        price_above_sma = close[i] > sma_200_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === WILLIAMS %R CROSS DETECTION ===
        # Cross above -80: oversold to neutral (reversal up)
        cross_above_80 = (wr > -80) and (wr_prev <= -80)
        # Cross below -20: overbought to neutral (reversal down)
        cross_below_20 = (wr < -20) and (wr_prev >= -20)
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG ENTRY: %R crosses above -80 with trend alignment ===
            if cross_above_80 and price_above_sma and vol_spike:
                desired_signal = SIZE
            
            # === SHORT ENTRY: %R crosses below -20 against trend ===
            if cross_below_20 and not price_above_sma and vol_spike:
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
        
        # === TAKE PROFIT: %R reaches opposite extreme ===
        tp_triggered = False
        
        if in_position and position_side > 0:
            # Long exit: %R reaches overbought
            if wr < -20:
                tp_triggered = True
        
        if in_position and position_side < 0:
            # Short exit: %R reaches oversold
            if wr > -80:
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