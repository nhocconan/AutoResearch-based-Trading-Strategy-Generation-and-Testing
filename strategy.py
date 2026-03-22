#!/usr/bin/env python3
"""
Experiment #004: 4h RSI-BB Mean Reversion with 12h HMA Trend Filter

Hypothesis: After 3 failed experiments with complex regime-switching and Connors RSI,
simplify to proven mean-reversion edge that works in bear/range markets (2022, 2025):

1. 12h HMA = stable trend direction filter (proven in best strategies)
2. 4h RSI(14) extremes = entry trigger (<35 long, >65 short)
3. Bollinger Band confirmation = price must touch/ breach bands
4. ATR volatility filter = skip dead chop (ATR ratio > 0.8)
5. Asymmetric bias = prefer longs when 12h bullish, smaller shorts

Why this should beat failed experiments:
- RSI+BB mean reversion works in bear/range (2022 crash, 2025 bear)
- 12h HMA filter prevents counter-trend disasters
- Simple = fewer conflicting conditions = MORE trades (need ≥10/symbol)
- ATR filter skips low-vol periods where most losses occur

Timeframe: 4h (REQUIRED for this experiment)
HTF: 12h via mtf_data helper (call ONCE before loop)
Position size: 0.25-0.30 discrete
Stoploss: 2.5 * ATR trailing
Target: 25-45 trades/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_rsi_bb_12h_hma_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    mid = sma.values
    
    return upper.values, lower.values, mid

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 12h HTF data ONCE before loop (CRITICAL - Rule 1)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h HMA for trend bias
    hma_12h = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, 20, 2.0)
    
    # ATR ratio for volatility filter (current vs 30-bar avg)
    atr_30 = calculate_atr(high, low, close, 30)
    atr_ratio = atr_14 / np.maximum(atr_30, 1e-10)
    
    # Volume SMA for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28
    
    # Track position for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    
    for i in range(100, n):
        # Skip if any indicator is NaN
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_12h_aligned[i]):
            continue
        if np.isnan(rsi_14[i]):
            continue
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        if np.isnan(atr_ratio[i]):
            continue
        
        # === 12H HMA TREND BIAS ===
        bull_bias = close[i] > hma_12h_aligned[i]
        bear_bias = close[i] < hma_12h_aligned[i]
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        
        # === BOLLINGER BAND TOUCH ===
        touch_bb_lower = low[i] <= bb_lower[i] * 1.002
        touch_bb_upper = high[i] >= bb_upper[i] * 0.998
        
        # === VOLATILITY FILTER ===
        vol_ok = atr_ratio[i] > 0.75
        
        # === VOLUME FILTER ===
        vol_confirmed = volume[i] > 0.6 * vol_sma[i] if not np.isnan(vol_sma[i]) else True
        
        # === POSITION SIZING ===
        size_mult = np.clip(1.0 / np.maximum(atr_ratio[i], 0.6), 0.7, 1.2)
        current_size = BASE_SIZE * size_mult
        current_size = np.clip(current_size, 0.20, 0.35)
        
        # Asymmetric: smaller shorts in bull market
        long_size = current_size
        short_size = current_size * 0.75 if bull_bias else current_size
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG: RSI oversold + BB lower touch + 12h not strongly bearish + vol ok
        if rsi_oversold and touch_bb_lower and vol_ok and vol_confirmed:
            if bull_bias or not bear_bias:  # Allow if neutral or bullish
                new_signal = long_size
        
        # SHORT: RSI overbought + BB upper touch + 12h not strongly bullish + vol ok
        if rsi_overbought and touch_bb_upper and vol_ok and vol_confirmed:
            if bear_bias or not bull_bias:  # Allow if neutral or bearish
                new_signal = -short_size
        
        # === STOPLOSS: 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stop_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stop_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stop_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stop_price:
                    stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position or np.sign(new_signal) != position_side:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
        
        signals[i] = new_signal
    
    return signals