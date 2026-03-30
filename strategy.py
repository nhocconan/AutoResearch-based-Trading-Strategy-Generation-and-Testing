#!/usr/bin/env python3
"""
Experiment #010: 1d Donchian(20) Breakout + Volume + RSI + 1w SMA Trend

HYPOTHESIS: Daily Donchian(20) breakouts capture institutional momentum moves.
RSI(14) < 30 filters oversold bounce entries, RSI > 70 filters overextended shorts.
Volume confirmation (1.8x) eliminates false breakouts. 1w SMA ensures larger trend
alignment. Works in both bull (upward breakouts) and bear (breakdown momentum).

WHY 1d: Each bar = significant price action. Filters noise. Fewer but higher-
quality trades. Target: 50-100 total over 4 years = 12-25/year.

ENTRIES:
- LONG: close > DonchianHigh(20) + RSI < 45 + vol > 1.8x + price > 1w SMA
- SHORT: close < DonchianLow(20) + RSI > 55 + vol > 1.8x + price < 1w SMA

TARGET: 50-100 total trades (12-25/year). HARD MAX: 150.
Signal size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_rsi_vol_1w_v1"
timeframe = "1d"
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

def calculate_rsi(prices, period=14):
    """RSI indicator"""
    delta = pd.Series(prices).diff()
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
    
    # === Load HTF data ONCE before loop ===
    df_1w = get_htf_data(prices, '1w')
    
    # 1w SMA for trend direction
    sma_1w = pd.Series(df_1w['close'].values).rolling(window=8, min_periods=8).mean().values
    sma_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_1w)
    
    # === Local 1d indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    
    # Donchian channels (shift by 1 to avoid look-ahead)
    donchian_high = pd.Series(high).rolling(20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(low).rolling(20, min_periods=20).min().shift(1).values
    
    # Volume ratio (20-bar average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(sma_1w_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Current Donchian levels
        dn_high = donchian_high[i]
        dn_low = donchian_low[i]
        
        if np.isnan(dn_high) or np.isnan(dn_low):
            signals[i] = 0.0
            continue
        
        # RSI value
        rsi_val = rsi_14[i]
        if np.isnan(rsi_val):
            signals[i] = 0.0
            continue
        
        # 1w SMA trend
        price_above_1w = close[i] > sma_1w_aligned[i]
        
        # Volume spike
        vol_spike = vol_ratio[i] > 1.8
        
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Breakout above 20d high + RSI not overbought + volume + trend up ===
            if close[i] > dn_high and rsi_val < 70 and vol_spike and price_above_1w:
                desired_signal = SIZE
            
            # === SHORT: Breakdown below 20d low + RSI not oversold + volume + trend down ===
            if close[i] < dn_low and rsi_val > 30 and vol_spike and not price_above_1w:
                desired_signal = -SIZE
        
        # === STOPLOSS: 2.5 ATR trailing stop ===
        if in_position:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
                trailing_stop = highest_since_entry - 2.5 * entry_atr
                if low[i] < trailing_stop:
                    desired_signal = 0.0
            
            if position_side < 0:
                lowest_since_entry = min(lowest_since_entry, low[i])
                trailing_stop = lowest_since_entry + 2.5 * entry_atr
                if high[i] > trailing_stop:
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
                entry_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals