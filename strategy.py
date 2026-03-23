#!/usr/bin/env python3
"""
Experiment #115: 1h Primary + 4h HTF — HMA Trend + RSI Pullback + Volume

Hypothesis: Lower TF strategies (105,108,110) failed with 0 trades due to over-filtering.
This uses LOOSE entry conditions to ensure trades while maintaining HTF trend alignment:

1) 4h HMA(21) for macro trend bias — only trade in HTF trend direction
2) 1h RSI(14) pullback — enter on pullback (RSI 40-60) not extremes
3) Volume > 0.8x 20-bar avg — basic confirmation (loose filter)
4) ATR(14) 2.5x trailing stop — risk management
5) NO session filter — too restrictive for 1h (caused 0 trades before)
6) NO Choppiness Index — adds lag, failed in previous experiments

Why this should work:
- Simpler = more trades (target 40-60/year on 1h)
- HTF trend filter prevents counter-trend trades in 2022 crash / 2025 bear
- RSI pullback (40-60, not 20/80 extremes) = many more entries than CRSI<10
- Proven pattern from best strategy (mtf_4h_crsi_chop_donchian_regime_1d1w_v3)
- Conservative size (0.25) limits drawdown during 2022 -77% crash

Position size: 0.25 (conservative for 1h, max 0.30 with volume confluence)
Stoploss: 2.5*ATR trailing
Target: 40-60 trades/year, Sharpe > 0.3 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_vol_pullback_4h_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

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
    rsi_14 = calculate_rsi(close, period=14)
    vol_avg_20 = calculate_volume_avg(volume, period=20)
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.25
    POSITION_SIZE_MAX = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_4h_aligned[i]) or np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(rsi_14[i]) or np.isnan(vol_avg_20[i]) or vol_avg_20[i] == 0:
            continue
        
        # === HTF TREND BIAS (4h HMA) ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        volume_ratio = volume[i] / (vol_avg_20[i] + 1e-10)
        volume_ok = volume_ratio > 0.8
        volume_strong = volume_ratio > 1.5
        
        # === ENTRY LOGIC (LOOSE CONDITIONS) ===
        new_signal = 0.0
        
        # --- LONG ENTRY ---
        # 4h trend up + RSI pullback (40-60, not extreme) + volume
        if price_above_hma_4h and volume_ok:
            if 40.0 <= rsi_14[i] <= 60.0:
                new_signal = POSITION_SIZE_BASE
                if volume_strong:
                    new_signal = POSITION_SIZE_MAX
        
        # --- SHORT ENTRY ---
        # 4h trend down + RSI pullback (40-60, not extreme) + volume
        if price_below_hma_4h and volume_ok:
            if 40.0 <= rsi_14[i] <= 60.0:
                new_signal = -POSITION_SIZE_BASE
                if volume_strong:
                    new_signal = -POSITION_SIZE_MAX
        
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
        
        # === EXIT ON RSI EXTREME (take profit) ===
        if in_position and position_side > 0 and rsi_14[i] > 70.0:
            new_signal = 0.0
        
        if in_position and position_side < 0 and rsi_14[i] < 30.0:
            new_signal = 0.0
        
        # === EXIT ON HTF TREND REVERSAL ===
        if in_position and position_side > 0 and price_below_hma_4h:
            new_signal = 0.0
        
        if in_position and position_side < 0 and price_above_hma_4h:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals