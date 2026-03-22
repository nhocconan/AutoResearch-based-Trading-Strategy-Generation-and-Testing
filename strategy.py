#!/usr/bin/env python3
"""
Experiment #342: 1d Vol Spike Mean Reversion with 1w HMA Trend Filter

Hypothesis: After 291 failed strategies, the key insight is that daily timeframe
needs fewer, higher-quality signals that capture volatility exhaustion events.
This strategy targets "capitulation bounces" - when volatility spikes AND price
reaches extreme Bollinger levels, signaling potential mean reversion.

Key Components:
1. VOLATILITY SPIKE DETECTION: ATR(7)/ATR(30) > 1.8 captures panic/extreme moves
2. BOLLINGER EXTREMES: Price below BB_lower (long) or above BB_upper (short)
3. 1W HMA TREND FILTER: Only long if price > 1w HMA, only short if price < 1w HMA
4. RSI CONFIRMATION: RSI(14) < 35 for longs, > 65 for shorts (loose thresholds)
5. ATR TRAILING STOP: 2.5x ATR to protect against continued trends

Why this should work on 1d:
- Daily bars filter out noise, capture meaningful volatility events
- Vol spike + BB extreme = high-probability mean reversion setup
- 1w HMA provides stable trend bias (proven in successful strategies)
- Loose RSI thresholds ensure >=10 trades per symbol on train period
- Fewer trades = less fee drag, higher win rate per trade

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_vol_spike_1w_hma_bb_rsi_meanrev_atr_v1"
timeframe = "1d"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate RSI indicator."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / np.maximum(avg_loss, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    return upper.values, lower.values, sma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr_7 = calculate_atr(high, low, close, 7)
    atr_14 = calculate_atr(high, low, close, 14)
    atr_30 = calculate_atr(high, low, close, 30)
    rsi = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(atr_7[i]) or np.isnan(atr_30[i]):
            signals[i] = 0.0
            continue
        
        if atr_30[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            continue
        
        # === VOLATILITY SPIKE DETECTION ===
        vol_ratio = atr_7[i] / max(atr_30[i], 1e-10)
        vol_spike = vol_ratio > 1.8  # Volatility 80% above normal
        
        # === BOLLINGER BAND POSITION ===
        price_below_bb = close[i] < bb_lower[i]
        price_above_bb = close[i] > bb_upper[i]
        
        # === 1W HMA TREND FILTER ===
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        # === RSI CONFIRMATION (loose thresholds for more trades) ===
        rsi_oversold = rsi[i] < 40  # Loosened from 35
        rsi_overbought = rsi[i] > 60  # Loosened from 65
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # LONG ENTRY: Vol spike + price below BB + bullish 1w trend + RSI oversold
        if vol_spike and price_below_bb and bull_trend_1w and rsi_oversold:
            new_signal = SIZE
        
        # SHORT ENTRY: Vol spike + price above BB + bearish 1w trend + RSI overbought
        elif vol_spike and price_above_bb and bear_trend_1w and rsi_overbought:
            new_signal = -SIZE
        
        # === SECONDARY ENTRY (no vol spike, just BB extreme + trend) ===
        # This ensures we get enough trades even when vol doesn't spike
        elif price_below_bb and bull_trend_1w and rsi_oversold:
            new_signal = SIZE * 0.5  # Half size for lower conviction
        
        elif price_above_bb and bear_trend_1w and rsi_overbought:
            new_signal = -SIZE * 0.5  # Half size for lower conviction
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit if 1w trend flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_1w:
                new_signal = 0.0
            if position_side < 0 and bull_trend_1w:
                new_signal = 0.0
        
        # === RSI EXIT (mean reversion complete) ===
        # Exit long when RSI crosses above 60, exit short when RSI crosses below 40
        if in_position and new_signal != 0.0:
            if position_side > 0 and rsi[i] > 60:
                new_signal = 0.0
            if position_side < 0 and rsi[i] < 40:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals