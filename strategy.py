#!/usr/bin/env python3
"""
Experiment #004: 1d Donchian Breakout + Volume + RSI + 1w Trend

HYPOTHESIS: 1d timeframe naturally limits trade frequency to ~15-20/year.
Donchian(20) breakout captures institutional moves. Volume spike confirms 
breakout validity. 1w SMA200 trend filter ensures we trade with the larger trend.
RSI(14) adds confirmation. ATR stoploss protects against whipsaws.

WHY THIS SHOULD WORK IN BOTH BULL AND BEAR:
- Breakout strategies work in both directions (long breakouts in bull, shorts in bear)
- 1d timeframe filters noise that destroys 4h strategies
- Tight stoploss prevents 2022 crash blowup
- Volume confirmation reduces false breakouts

TARGET: 60-100 total trades over 4 years (15-25/year).
DB reference: mtf_4h_hma_donchian_volume_rsi_12h_atr_v1 (Sharpe=1.382)

KEY DESIGN (minimal conditions = fewer trades = less fee drag):
1. Donchian(20) breakout on 1d
2. Volume > 1.5x 20-day MA confirmation
3. RSI(14) filter (not extreme, just directional)
4. 1w SMA200 trend filter
5. ATR(14) stoploss at 2x
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_vol_rsi_1w_v1"
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

def calculate_rsi(close, period=14):
    """RSI indicator"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    deltas = np.diff(close, prepend=close[0])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    
    avg_gain = pd.Series(gains).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(losses).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = avg_gain / np.where(avg_loss > 0, avg_loss, 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def calculate_sma(values, period):
    """Simple Moving Average"""
    return pd.Series(values).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # 1w SMA200 for trend direction
    sma_200_1w = calculate_sma(df_1w['close'].values, 200)
    sma_200_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_200_1w)
    
    # Calculate 1d Donchian channels
    period_donchian = 20
    donchian_high = pd.Series(high).rolling(window=period_donchian, min_periods=period_donchian).max().values
    donchian_low = pd.Series(low).rolling(window=period_donchian, min_periods=period_donchian).min().values
    
    # ATR for stoploss
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # RSI
    rsi_14 = calculate_rsi(close, period=14)
    
    # Volume MA
    vol_ma = calculate_sma(volume, 20)
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 250  # Need enough for SMA200 1w alignment
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(sma_200_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND FILTER (1w SMA200) ===
        price_above_1w_sma = close[i] > sma_200_1w_aligned[i]
        
        # === DONCHIAN BREAKOUT ===
        donch_high = donchian_high[i]
        donch_low = donchian_low[i]
        
        if np.isnan(donch_high) or np.isnan(donch_low):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === RSI DIRECTION ===
        rsi = rsi_14[i]
        rsi_bullish = rsi < 60
        rsi_bearish = rsi > 40
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: Break above 20-day high + bullish conditions
        if not in_position or position_side <= 0:
            # Price breaks above Donchian high
            breakout_long = close[i] > donch_high
            
            # Must have: volume spike, RSI not overbought, above 1w SMA
            if breakout_long and vol_spike and rsi_bullish and price_above_1w_sma:
                desired_signal = SIZE
        
        # SHORT: Break below 20-day low + bearish conditions
        if not in_position or position_side >= 0:
            # Price breaks below Donchian low
            breakout_short = close[i] < donch_low
            
            # Must have: volume spike, RSI not oversold, below 1w SMA
            if breakout_short and vol_spike and rsi_bearish and not price_above_1w_sma:
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2x ATR) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = close[i] - 2.0 * entry_atr
                else:
                    stop_price = close[i] + 2.0 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = desired_signal
    
    return signals