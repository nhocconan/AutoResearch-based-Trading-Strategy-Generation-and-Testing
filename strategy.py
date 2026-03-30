#!/usr/bin/env python3
"""
Experiment #021: 12h Donchian Breakout + 1d HMA Trend + Volume Confirmation

HYPOTHESIS: 12h is the sweet spot between 4h (too many trades) and 1d (too few).
Combining:
1. 12h Donchian(20) breakout - proven price channel structure from DB winners
2. 1d HMA(21) trend filter - removes entries against primary trend
3. Volume spike confirmation - validates breakout strength
4. ATR stoploss - 2.5x ATR risk management

WHY IT SHOULD WORK IN BOTH BULL AND BEAR:
- Bull: Price breaks 12h high + above 1d HMA = continuation long, volume confirms
- Bear: Price breaks 12h low + below 1d HMA = continuation short, volume confirms
- Range: No breakout = no trade (avoids whipsaws at tops/bottoms)
- 12h gives enough bars for meaningful breakouts without overtrading

TARGET: 100-200 total trades over 4 years (25-50/year on 12h = achievable)
LEARNED FROM FAILURES:
- Ichimoku/Alligator too complex → 175 trades, barely broke even
- Simple Donchian + HMA + volume is the proven winning formula
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_1d_hma_vol_v3"
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

def calculate_hma(data, period):
    """Hull Moving Average"""
    half = int(period / 2)
    sqrt_period = int(np.sqrt(period))
    
    wma_half = pd.Series(data).rolling(window=half, min_periods=half).apply(
        lambda x: np.sum(x * np.arange(1, len(x)+1)) / np.sum(np.arange(1, len(x)+1)), raw=True
    ).values
    wma_full = pd.Series(data).rolling(window=period, min_periods=period).apply(
        lambda x: np.sum(x * np.arange(1, len(x)+1)) / np.sum(np.arange(1, len(x)+1)), raw=True
    ).values
    
    hma = 2 * wma_half - wma_full
    return hma

def calculate_donchian(high, low, period=20):
    """Donchian Channel - rolling high/low"""
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d HMA for trend direction ===
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    donchian_high, donchian_low = calculate_donchian(high, low, period=20)
    
    # Volume ratio (20-period MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    
    warmup = 100  # 20 Donchian + 14 ATR + 20 volume MA
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === HTF TREND (1d HMA) ===
        price_above_hma = close[i] > hma_1d_aligned[i]
        price_below_hma = close[i] < hma_1d_aligned[i]
        
        # === DONCHIAN BREAKOUT ===
        # Breakout: price closes above/below 20-bar high/low
        bullish_breakout = close[i] > donchian_high[i] and close[i-1] <= donchian_high[i-1]
        bearish_breakout = close[i] < donchian_low[i] and close[i-1] >= donchian_low[i-1]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG: Price breaks 12h high + above 1d HMA + volume confirms
            if bullish_breakout and price_above_hma and vol_spike:
                desired_signal = SIZE
            
            # SHORT: Price breaks 12h low + below 1d HMA + volume confirms
            elif bearish_breakout and price_below_hma and vol_spike:
                desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR) ===
        if in_position:
            if position_side > 0:
                # Long stop: entry - 2.5 ATR
                stop_price = entry_price - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Exit if price falls back below 1d HMA (trend reversal)
                if price_below_hma:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Short stop: entry + 2.5 ATR
                stop_price = entry_price + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Exit if price rises back above 1d HMA (trend reversal)
                if price_above_hma:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 2 bars to reduce fee churn ===
        if in_position and (i - entry_bar) < 2:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals