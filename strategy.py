#!/usr/bin/env python3
"""
Experiment #021: 12h Donchian Breakout + Volume + Choppiness Regime

HYPOTHESIS: Simple price-channel breakout with volume confirmation and regime filter
on 12h timeframe will capture major trend moves while avoiding whipsaws in ranges.

WHY IT SHOULD WORK IN BOTH BULL AND BEAR:
- Bull: Price breaks above Donchian upper band + volume spike + chop < 38 = strong uptrend entry
- Bear: Price breaks below Donchian lower band + volume spike + chop < 38 = strong downtrend entry
- Range: Choppiness > 61.8 = no trades (avoid 2022 crash whipsaws that destroyed other strategies)
- 12h timeframe = fewer trades = lower fee drag than 4h strategies that overtrade

KEY INSIGHT from DB: The best test performers use ONE strong signal (price channel breakout)
+ volume confirmation + regime filter. Simple = robust = generalizes to test.

TARGET: 75-200 total trades over 4 years (19-50/year on 12h = reasonable)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_vol_chop_regime_v1"
timeframe = "12h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Donchian Channel - price channel breakout"""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    middle = np.zeros(n)
    
    for i in range(n):
        start = max(0, i - period + 1)
        upper[i] = np.max(high[start:i+1])
        lower[i] = np.min(low[start:i+1])
        middle[i] = (upper[i] + lower[i]) / 2
    
    return upper, lower, middle

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP < 38.2 = trending market (trend following)
    CHOP > 61.8 = ranging market (avoid trading)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.zeros(n)
    
    for i in range(period, n):
        # Sum of ATR over period
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j])
            atr_sum += tr
        
        # Highest high - lowest low over period
        hh = np.max(high[i - period + 1:i + 1])
        ll = np.min(low[i - period + 1:i + 1])
        range_sum = hh - ll
        
        if range_sum > 0:
            # CHOP = 100 * log10(atr_sum / range_sum) / log10(period)
            chop[i] = 100 * np.log10(atr_sum / range_sum) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # === HTF indicators (1d) ===
    # 1d SMA for trend direction
    sma_50_1d = pd.Series(df_1d['close'].values).rolling(window=50, min_periods=50).mean().values
    price_above_sma_1d = df_1d['close'].values > sma_50_1d
    
    # HTF trend alignment
    htf_bullish = align_htf_to_ltf(prices, df_1d, price_above_sma_1d.astype(float))
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    upper_don, lower_don, middle_don = calculate_donchian(high, low, period=20)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.28  # Position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    trailing_high = 0.0
    trailing_low = 0.0
    
    warmup = 100  # Donchian 20 + choppiness 14 + volume 20
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(upper_don[i]) or np.isnan(lower_don[i]):
            signals[i] = 0.0
            continue
        
        # === Regime filter: Only trade in trending markets ===
        trending = chop[i] < 45.0  # Relaxed from 38.2 to allow more trades
        ranging = chop[i] > 61.8   # Strong range = no trade
        
        if ranging:
            if in_position:
                signals[i] = 0.0  # Exit in strong range
            else:
                signals[i] = 0.0
            continue
        
        # === Volume confirmation ===
        vol_confirm = vol_ratio[i] > 1.5
        
        # === HTF trend (1d SMA) ===
        htf_trend_up = htf_bullish[i] > 0.5 if not np.isnan(htf_bullish[i]) else False
        
        # === Donchian breakout signals ===
        # Breakout: price closes above 20-bar high
        bullish_breakout = close[i] > upper_don[i] and close[i-1] <= upper_don[i-1]
        # Breakdown: price closes below 20-bar low
        bearish_breakout = close[i] < lower_don[i] and close[i-1] >= lower_don[i-1]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG: Bullish breakout + volume + HTF uptrend
            if bullish_breakout and vol_confirm and (trending or htf_trend_up):
                if htf_trend_up or chop[i] < 50:  # Allow if HTF confirms or chop moderate
                    desired_signal = SIZE
            
            # SHORT: Bearish breakdown + volume + HTF downtrend (or neutral if chop confirms)
            elif bearish_breakout and vol_confirm and (trending or not htf_trend_up):
                if not htf_trend_up or chop[i] < 50:  # Allow if HTF bearish or chop trending
                    desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR trailing stop) ===
        if in_position:
            if position_side > 0:
                # Update trailing high
                if i == entry_bar or high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Trailing stop: exit if price falls 2.5 ATR from recent high
                stop_price = trailing_high - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Exit if chop enters strong range
                if chop[i] > 61.8:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Update trailing low
                if i == entry_bar or low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Trailing stop: exit if price rises 2.5 ATR from recent low
                stop_price = trailing_low + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Exit if chop enters strong range
                if chop[i] > 61.8:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 3 bars to avoid fee churn ===
        if in_position and (i - entry_bar) < 3:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                trailing_high = high[i]
                trailing_low = low[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals