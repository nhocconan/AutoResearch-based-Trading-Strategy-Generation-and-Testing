#!/usr/bin/env python3
"""
Experiment #158: 30m Primary + 4h/1d HTF — Fisher Transform Reversal Strategy

Hypothesis: CRSI-based strategies have been tried 13+ times and keep failing (negative Sharpe).
The Fisher Transform is fundamentally different - it normalizes price to Gaussian distribution,
making extreme reversals clearer than RSI. Proven in bear/range markets (2022 crash, 2025 bear).

Key differences from failed #148 (0 trades):
1) Fisher Transform(9) instead of RSI/CRSI - better reversal detection at extremes
2) Entry: Fisher crosses -1.5 (long) or +1.5 (short) - clearer than CRSI thresholds
3) Hold logic: stay in position until Fisher crosses opposite threshold (reduces churn)
4) Volume filter: 1.2x avg (not 1.5x) - allows more valid entries
5) Position size: 0.20 base, 0.30 with full confluence (conservative for 30m)

MTF Structure:
- 4h HMA(21): trend direction bias (trade WITH 4h trend)
- 1d HMA(21): macro confirmation (adds size when aligned)
- 30m Fisher(9): entry timing within HTF trend
- 30m Volume(20): confirmation filter (>1.2x average)
- 30m ATR(14): stoploss at 2.5x (mandatory risk management)

Target: 40-80 trades/year, Sharpe > 0.5 on ALL symbols (BTC, ETH, SOL)
Position size: 0.20 base, 0.30 max (smaller for lower TF to reduce fee impact)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_fisher_reversal_hma_4h1d_v1"
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

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution.
    Makes extreme reversals clearer than RSI.
    
    Formula:
    1. Price = (2 * (close - lowest_low) / (highest_high - lowest_low) - 1)
    2. Price smoothed with EMA
    3. Fisher = 0.5 * ln((1 + Price) / (1 - Price))
    
    Entry signals:
    - Long: Fisher crosses above -1.5 from below
    - Short: Fisher crosses below +1.5 from above
    """
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    
    # Avoid division by zero
    price = np.zeros(len(close))
    mask = price_range > 0
    price[mask] = 2.0 * (close[mask] - lowest_low[mask]) / price_range[mask] - 1.0
    
    # Clip to avoid ln domain errors
    price = np.clip(price, -0.999, 0.999)
    
    # Smooth with EMA
    price_s = pd.Series(price)
    price_smooth = price_s.ewm(span=period, min_periods=period, adjust=False).mean().values
    price_smooth = np.clip(price_smooth, -0.999, 0.999)
    
    # Fisher transform
    fisher = 0.5 * np.log((1.0 + price_smooth) / (1.0 - price_smooth))
    
    return fisher

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h HMA for trend direction
    hma_4h = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1d HMA for macro confirmation
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    fisher = calculate_fisher_transform(high, low, close, period=9)
    
    # Volume average (20-bar)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.20
    POSITION_SIZE_MAX = 0.30
    
    # Track position state for stoploss and hold logic
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_fisher = 0.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(fisher[i]):
            continue
        if np.isnan(vol_avg[i]) or vol_avg[i] == 0:
            continue
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 1.2 * vol_avg[i]
        
        # === HTF TREND BIAS ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_cross_up_long = False
        fisher_cross_down_short = False
        
        if i > 0 and not np.isnan(fisher[i-1]):
            # Long: Fisher crosses above -1.5 from below
            if fisher[i-1] < -1.5 and fisher[i] >= -1.5:
                fisher_cross_up_long = True
            # Short: Fisher crosses below +1.5 from above
            if fisher[i-1] > 1.5 and fisher[i] <= 1.5:
                fisher_cross_down_short = True
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY ---
        # Fisher reversal + 4h trend aligned + volume confirmation
        if fisher_cross_up_long and volume_ok:
            if price_above_hma_4h:
                new_signal = POSITION_SIZE_BASE
                # Add size if 1d also aligned
                if price_above_hma_1d:
                    new_signal = POSITION_SIZE_MAX
        
        # --- SHORT ENTRY ---
        # Fisher reversal + 4h trend aligned + volume confirmation
        if fisher_cross_down_short and volume_ok:
            if price_below_hma_4h:
                new_signal = -POSITION_SIZE_BASE
                # Add size if 1d also aligned
                if price_below_hma_1d:
                    new_signal = -POSITION_SIZE_MAX
        
        # === HOLD POSITION LOGIC ===
        # Stay in position until Fisher crosses opposite threshold OR stoploss
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long until Fisher > 1.5 (overbought exit)
                if fisher[i] < 1.5:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short until Fisher < -1.5 (oversold exit)
                if fisher[i] > -1.5:
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
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
                entry_fisher = fisher[i]
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
                entry_fisher = fisher[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                entry_fisher = 0.0
        
        signals[i] = new_signal
    
    return signals