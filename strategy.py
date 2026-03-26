#!/usr/bin/env python3
"""
Experiment #012: 12h ATR Channel Breakout + 1d Trend + Volume

HYPOTHESIS: ATR-based channels are more responsive than Donchian and adapt 
to volatility. Price closing beyond EMA +/- 1.5*ATR marks institutional moves.
Combined with 1d trend (SMA200) and volume confirmation, this captures major 
breakouts while filtering noise. Works in both directions.

Key insight: ATR channels adapt to volatility - in low vol periods channels 
are tight (more breakouts), in high vol periods channels widen (fewer breakouts).
This self-adjusting property prevents overtrading.

TIMEFRAME: 12h primary
HTF: 1d for trend bias
TARGET: 75-200 total trades over 4 years (19-50/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_atr_channel_1d_sma200_v1"
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
        tr[i] = max(high[i] - low[i], 
                    abs(high[i] - close[i-1]), 
                    abs(low[i] - close[i-1]))
    
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
    
    # 1d SMA200 for trend bias (align to 12h)
    sma200_1d = df_1d['close'].values
    # Need to compute rolling mean properly
    sma200_1d_vals = pd.Series(sma200_1d).rolling(window=200, min_periods=200).mean().values
    sma200_1d_aligned = align_htf_to_ltf(prices, df_1d, sma200_1d_vals)
    
    # Calculate local 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # EMA 21 for channel center
    ema_21 = pd.Series(close).ewm(span=21, min_periods=21, adjust=False).mean().values
    
    # ATR Channel bands
    atr_mult = 1.5
    upper_band = ema_21 + atr_mult * atr_14
    lower_band = ema_21 - atr_mult * atr_14
    
    # Volume MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    stop_price = 0.0
    
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(ema_21[i]) or np.isnan(sma200_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND BIAS (1d SMA200) ===
        price_above_1d_sma = close[i] > sma200_1d_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.4
        
        # === CHANNEL STATUS ===
        price_above_upper = close[i] > upper_band[i]
        price_below_lower = close[i] < lower_band[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === NEW LONG ENTRY ===
            # Price closes above upper ATR band + volume spike + bullish trend
            if price_above_upper and vol_spike and price_above_1d_sma:
                desired_signal = SIZE
            
            # === NEW SHORT ENTRY ===
            # Price closes below lower ATR band + volume spike + bearish trend
            if price_below_lower and vol_spike and not price_above_1d_sma:
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
        
        # === TAKE PROFIT: 2R partial exit ===
        if in_position and position_side > 0:
            profit_2r = (close[i] - entry_price) >= 2.0 * entry_atr
            if profit_2r:
                desired_signal = SIZE / 2  # Half position
        
        if in_position and position_side < 0:
            profit_2r = (entry_price - close[i]) >= 2.0 * entry_atr
            if profit_2r:
                desired_signal = -SIZE / 2  # Half position
        
        # === EXIT: Price returns inside channel ===
        if in_position and position_side > 0:
            # Long exit: price falls back below upper band
            if not price_above_upper and close[i] < upper_band[i]:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Short exit: price rises back above lower band
            if not price_below_lower and close[i] > lower_band[i]:
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
                # Same direction - maintain or adjust size
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
                entry_bar = 0
        
        signals[i] = desired_signal
    
    return signals