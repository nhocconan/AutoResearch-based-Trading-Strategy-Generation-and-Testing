#!/usr/bin/env python3
"""
Experiment #120: 1h Primary + 4h HTF — HMA Trend + RSI Pullback + BB Confluence

Hypothesis: Lower timeframe strategies fail due to either (1) too many trades causing fee drag,
or (2) too strict filters causing 0 trades. This strategy balances by:

1) 4h HMA(21) for macro trend bias — only trade pullbacks in trend direction
2) 1h RSI(7) for entry timing — enter on oversold (RSI<35) in uptrend, overbought (RSI>65) in downtrend
3) 1h Bollinger Band position — price must be near opposite band (mean reversion within trend)
4) Volume confirmation — volume > 0.8x 20-period average (filters low-liquidity entries)
5) ATR(14) trailing stop at 2.5x — protects capital, locks profits

Why this should work on 1h:
- 4h trend filter reduces whipsaws (HTF direction, LTF entry)
- RSI(7) is faster than RSI(14), catches pullbacks earlier
- BB confluence ensures we're buying dips in uptrend, not chasing
- Volume filter avoids low-liquidity false signals
- 1h naturally produces 40-80 trades/year with these filters (acceptable fee drag)

Position size: 0.25 base (conservative for 1h TF)
Stoploss: 2.5*ATR trailing
Target: 40-80 trades/year, Sharpe > 0.5 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_bb_pullback_4h_v1"
timeframe = "1h"
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
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    gain = np.maximum(delta, 0)
    loss = -np.minimum(delta, 0)
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def calculate_volume_avg(volume, period=20):
    """Calculate rolling average volume."""
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_avg

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h HMA for macro trend
    hma_4h = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)
    rsi_14 = calculate_rsi(close, period=14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, period=20, std_mult=2.0)
    vol_avg_20 = calculate_volume_avg(volume, period=20)
    hma_1h_21 = calculate_hma(close, period=21)
    hma_1h_50 = calculate_hma(close, period=50)
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.25
    POSITION_SIZE_MAX = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_4h_aligned[i]) or np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(rsi_7[i]) or np.isnan(rsi_14[i]):
            continue
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(bb_mid[i]):
            continue
        if np.isnan(vol_avg_20[i]) or vol_avg_20[i] == 0:
            continue
        if np.isnan(hma_1h_21[i]) or np.isnan(hma_1h_50[i]):
            continue
        
        # === HTF TREND BIAS (4h HMA) ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === 1h TREND FILTER ===
        hma_1h_bullish = hma_1h_21[i] > hma_1h_50[i]
        hma_1h_bearish = hma_1h_21[i] < hma_1h_50[i]
        
        # === RSI PULLBACK SIGNALS ===
        rsi_oversold = rsi_7[i] < 35.0
        rsi_overbought = rsi_7[i] > 65.0
        rsi_extreme_low = rsi_7[i] < 25.0
        rsi_extreme_high = rsi_7[i] > 75.0
        
        # === BOLLINGER BAND POSITION ===
        bb_position = (close[i] - bb_lower[i]) / (bb_upper[i] - bb_lower[i] + 1e-10)
        near_lower_band = bb_position < 0.25  # Price in bottom 25% of BB
        near_upper_band = bb_position > 0.75  # Price in top 25% of BB
        
        # === VOLUME CONFIRMATION ===
        volume_ratio = volume[i] / (vol_avg_20[i] + 1e-10)
        volume_confirmed = volume_ratio > 0.8
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY ---
        # Require: 4h trend up + 1h trend up + RSI pullback oversold + near lower BB + volume
        if price_above_hma_4h and hma_1h_bullish:
            if rsi_oversold and near_lower_band and volume_confirmed:
                new_signal = POSITION_SIZE
                if rsi_extreme_low:
                    new_signal = POSITION_SIZE_MAX
        
        # --- SHORT ENTRY ---
        # Require: 4h trend down + 1h trend down + RSI pullback overbought + near upper BB + volume
        if price_below_hma_4h and hma_1h_bearish:
            if rsi_overbought and near_upper_band and volume_confirmed:
                new_signal = -POSITION_SIZE
                if rsi_extreme_high:
                    new_signal = -POSITION_SIZE_MAX
        
        # === HOLD POSITION LOGIC ===
        # Hold long if still in uptrend and not at extreme overbought
        if in_position and new_signal == 0.0:
            if position_side > 0:
                if price_above_hma_4h and hma_1h_bullish and rsi_14[i] < 75.0:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                if price_below_hma_4h and hma_1h_bearish and rsi_14[i] > 25.0:
                    new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON TREND REVERSAL ===
        if in_position and position_side > 0:
            if price_below_hma_4h or (hma_1h_bearish and rsi_14[i] > 70.0):
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if price_above_hma_4h or (hma_1h_bullish and rsi_14[i] < 30.0):
                new_signal = 0.0
        
        # === TAKE PROFIT AT BB OPPOSITE BAND ===
        if in_position and position_side > 0:
            if near_upper_band and rsi_14[i] > 65.0:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if near_lower_band and rsi_14[i] < 35.0:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals