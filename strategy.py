#!/usr/bin/env python3
"""
Experiment #138: 30m Primary + 4h/1d HTF — HMA Trend + RSI Pullback (Simplified)

Hypothesis: Recent failures (#128, #130, #131, #132, #135) all had Sharpe=0.000 
because of OVER-FILTERING — CRSI + CHOP + Session filters together = ZERO trades.

This strategy SIMPLIFIES entry logic while keeping MTF structure:
1) 4h HMA(21) for trend direction (proven in best strategies)
2) 30m RSI(14) for pullback entries (simpler than CRSI, more trades)
3) Choppiness Index as PERMISSIVE filter (CHOP < 60 allows trend trades)
4) NO session filter (killed trade count in previous experiments)
5) 1d HMA as secondary trend confirmation (not required, just boosts size)

Why this should work on 30m:
- 4h trend filter reduces whipsaws (HTF direction)
- RSI(14) < 35 / > 65 triggers ~50-80 trades/year (not too many, not zero)
- Permissive CHOP filter (< 60 vs < 45) allows more trend trades
- Simple logic = robust across BTC/ETH/SOL
- Position size 0.25 base keeps DD controlled

Target: 40-80 trades/year, Sharpe > 0.5 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_hma_rsi_pullback_chop_4h1d_v1"
timeframe = "30m"
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

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index (CHOP)."""
    # CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_slope(values):
    """Calculate percentage slope of array."""
    n = len(values)
    slope = np.zeros(n)
    for i in range(1, n):
        if not np.isnan(values[i]) and not np.isnan(values[i-1]) and values[i-1] != 0:
            slope[i] = (values[i] - values[i-1]) / values[i-1] * 100
        else:
            slope[i] = 0.0
    return slope

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h HMA for primary trend
    hma_4h = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    hma_4h_slope = calculate_slope(hma_4h_aligned)
    
    # Calculate 1d HMA for secondary trend confirmation
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1d_slope = calculate_slope(hma_1d_aligned)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    hma_30m_21 = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.25
    POSITION_SIZE_MAX = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]):
            continue
        if np.isnan(hma_30m_21[i]):
            continue
        
        # === HTF TREND BIAS (4h HMA) ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        hma_4h_slope_positive = hma_4h_slope[i] > 0.3
        hma_4h_slope_negative = hma_4h_slope[i] < -0.3
        
        # === 1d TREND CONFIRMATION (optional, boosts size) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        hma_1d_slope_positive = hma_1d_slope[i] > 0.3
        hma_1d_slope_negative = hma_1d_slope[i] < -0.3
        
        # === CHOPPININESS REGIME FILTER (PERMISSIVE) ===
        # CHOP < 60 = trending (allow trend trades)
        # CHOP >= 60 = ranging (skip trend trades, could do mean revert)
        chop_trending = chop_14[i] < 60.0
        
        # === RSI PULLBACK ENTRY ===
        # Long: RSI < 35 (oversold pullback in uptrend)
        # Short: RSI > 65 (overbought pullback in downtrend)
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY ---
        # Require: 4h trend up + RSI oversold + CHOP trending (permissive)
        if price_above_hma_4h and hma_4h_slope_positive:
            if rsi_oversold and chop_trending:
                new_signal = POSITION_SIZE_BASE
                # Boost size if 1d also confirms
                if price_above_hma_1d and hma_1d_slope_positive:
                    new_signal = POSITION_SIZE_MAX
        
        # --- SHORT ENTRY ---
        # Require: 4h trend down + RSI overbought + CHOP trending (permissive)
        if price_below_hma_4h and hma_4h_slope_negative:
            if rsi_overbought and chop_trending:
                new_signal = -POSITION_SIZE_BASE
                # Boost size if 1d also confirms
                if price_below_hma_1d and hma_1d_slope_negative:
                    new_signal = -POSITION_SIZE_MAX
        
        # === HOLD POSITION LOGIC ===
        # Hold if trend still intact (no need to exit on every RSI cross)
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if 4h trend still up
                if price_above_hma_4h:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if 4h trend still down
                if price_below_hma_4h:
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
            if price_below_hma_4h and hma_4h_slope_negative:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if price_above_hma_4h and hma_4h_slope_positive:
                new_signal = 0.0
        
        # === TAKE PROFIT ON RSI EXTREME ===
        if in_position and position_side > 0 and rsi_14[i] > 75.0:
            new_signal = 0.0
        
        if in_position and position_side < 0 and rsi_14[i] < 25.0:
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