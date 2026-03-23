#!/usr/bin/env python3
"""
Experiment #161: 4h Primary + 1d/1w HTF — Volatility Spike Mean Reversion with Trend Filter

Hypothesis: Previous 4h strategies failed because they were either pure trend (whipsawed in 2022/2025)
or pure mean-reversion (got crushed in strong trends). This strategy combines:

1) Volatility Spike Detection: ATR(7)/ATR(30) > 1.8 signals panic/extreme moves that revert
2) HTF Trend Filter: 1d HMA(21) + 1w HMA(21) for macro direction bias
3) Donchian(20) Breakout: Confirms momentum is real, not noise
4) RSI(14) Extremes: Entry timing on oversold/overbought within trend
5) Asymmetric Logic: Different thresholds for long vs short (bear market bias)
6) ATR(14) 2.5x Stoploss: Mandatory risk management

Why this should work:
- Vol spikes mark capitulation points (2022 bottom, 2025 dips) - mean revert opportunity
- HTF trend filter prevents fighting macro direction (no long in 1d downtrend)
- 4h timeframe naturally limits to 20-50 trades/year (fee drag controlled)
- Asymmetric thresholds account for crypto's long bias with short squeezes

Target: 25-45 trades/year per symbol, Sharpe > 0.5 on ALL symbols
Position size: 0.25 base, 0.35 with full confluence
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_vol_spike_reversion_1d1w_v1"
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
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper and lower)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_vol_ratio(atr_short, atr_long):
    """Calculate volatility spike ratio (short ATR / long ATR)."""
    ratio = atr_short / (atr_long + 1e-10)
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d HMA for medium-term trend
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 1w HMA for long-term trend
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 4h indicators
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_30 = calculate_atr(high, low, close, period=30)
    rsi_14 = calculate_rsi(close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Volatility spike ratio
    vol_ratio = calculate_vol_ratio(atr_7, atr_30)
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.25
    POSITION_SIZE_MAX = 0.35
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(vol_ratio[i]) or np.isnan(rsi_14[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === HTF TREND BIAS ===
        # 1d trend
        bullish_1d = close[i] > hma_1d_aligned[i]
        bearish_1d = close[i] < hma_1d_aligned[i]
        
        # 1w trend (stronger filter)
        bullish_1w = close[i] > hma_1w_aligned[i]
        bearish_1w = close[i] < hma_1w_aligned[i]
        
        # === VOLATILITY REGIME ===
        vol_spike = vol_ratio[i] > 1.8  # ATR(7) > 1.8x ATR(30) = panic/extreme
        vol_normal = vol_ratio[i] < 1.3  # Back to normal
        
        # === DONCHIAN POSITION ===
        near_donchian_high = close[i] > donchian_upper[i] * 0.98  # Within 2% of high
        near_donchian_low = close[i] < donchian_lower[i] * 1.02   # Within 2% of low
        broke_donchian_high = close[i] > donchian_upper[i]
        broke_donchian_low = close[i] < donchian_lower[i]
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        rsi_extreme_low = rsi_14[i] < 25.0
        rsi_extreme_high = rsi_14[i] > 75.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY: Vol spike + oversold + HTF bullish bias ---
        # Primary: Vol spike + RSI oversold + 1d bullish (1w neutral ok)
        if vol_spike and rsi_oversold and bullish_1d:
            # Add size if 1w also bullish or near Donchian low (capitulation)
            if bullish_1w or near_donchian_low:
                new_signal = POSITION_SIZE_MAX
            else:
                new_signal = POSITION_SIZE_BASE
        
        # Secondary: RSI extreme low + 1d & 1w bullish (strong confluence)
        elif rsi_extreme_low and bullish_1d and bullish_1w:
            new_signal = POSITION_SIZE_MAX
        
        # --- SHORT ENTRY: Vol spike + overbought + HTF bearish bias ---
        # Primary: Vol spike + RSI overbought + 1d bearish (1w neutral ok)
        if vol_spike and rsi_overbought and bearish_1d:
            # Add size if 1w also bearish or near Donchian high
            if bearish_1w or near_donchian_high:
                new_signal = -POSITION_SIZE_MAX
            else:
                new_signal = -POSITION_SIZE_BASE
        
        # Secondary: RSI extreme high + 1d & 1w bearish (strong confluence)
        elif rsi_extreme_high and bearish_1d and bearish_1w:
            new_signal = -POSITION_SIZE_BASE
        
        # === EXIT ON VOLATILITY NORMALIZATION ===
        # If in position and vol returns to normal, consider reducing
        if in_position and vol_normal:
            if position_side > 0 and rsi_14[i] > 55.0:
                # Long position, RSI recovered, reduce/exit
                new_signal = 0.0
            elif position_side < 0 and rsi_14[i] < 45.0:
                # Short position, RSI recovered, reduce/exit
                new_signal = 0.0
        
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