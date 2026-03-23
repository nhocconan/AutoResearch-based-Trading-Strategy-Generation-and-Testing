#!/usr/bin/env python3
"""
Experiment #163: 1d Primary + 1w HTF — Simplified Donchian Breakout Strategy

Hypothesis: Previous strategies failed due to over-filtering (CRSI + CHOP + multiple HTF).
This strategy simplifies to proven components:

1) 1w HMA(21) for macro trend — trade WITH weekly trend only
2) Donchian(20) breakout for entries — clean price action signal
3) ATR(14) stoploss at 2.5x — mandatory risk management
4) Volume filter — volume > 0.8x 20-bar avg (light filter)
5) NO CRSI, NO Choppiness — these have failed consistently on BTC/ETH

Key insight: 1d timeframe naturally limits trades to 20-50/year (low fee drag).
Simpler logic = more reliable signals across all symbols (BTC/ETH/SOL).

Why this might work:
- Donchian breakouts catch sustained moves (not whipsaws)
- 1w HMA filters counter-trend trades (major cause of losses in 2022)
- 1d timeframe = fewer trades = less fee drag
- No mean reversion (CRSI) which fails in strong trends

Target: 20-50 trades/year per symbol, Sharpe > 0.5 on ALL symbols
Position size: 0.30 base (conservative for daily timeframe)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_hma_1w_v1"
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
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel upper and lower bands."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA for macro trend direction
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Volume average (20-bar)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Price momentum (ROC 10)
    roc_10 = np.zeros(n)
    for i in range(10, n):
        if close[i-10] != 0:
            roc_10[i] = (close[i] - close[i-10]) / close[i-10] * 100
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_1w_aligned[i]) or hma_1w_aligned[i] == 0:
            continue
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(vol_avg[i]) or vol_avg[i] == 0:
            continue
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * vol_avg[i]
        
        # === HTF TREND BIAS (1w HMA) ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === DONCHIAN BREAKOUT DETECTION ===
        # Breakout above upper band
        breakout_long = close[i] > donchian_upper[i-1] and close[i-1] <= donchian_upper[i-1]
        # Breakout below lower band
        breakout_short = close[i] < donchian_lower[i-1] and close[i-1] >= donchian_lower[i-1]
        
        # === MOMENTUM CONFIRMATION ===
        momentum_positive = roc_10[i] > 0
        momentum_negative = roc_10[i] < 0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # Long: Donchian breakout + price above 1w HMA + volume + momentum
        if breakout_long and price_above_hma_1w and volume_ok and momentum_positive:
            new_signal = POSITION_SIZE
        
        # Short: Donchian breakout + price below 1w HMA + volume + momentum
        if breakout_short and price_below_hma_1w and volume_ok and momentum_negative:
            new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        # Hold if in position and no exit signal
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if price still above 1w HMA
                if price_above_hma_1w:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if price still below 1w HMA
                if price_below_hma_1w:
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
        # Exit long if price crosses below 1w HMA
        if in_position and position_side > 0 and price_below_hma_1w:
            new_signal = 0.0
        
        # Exit short if price crosses above 1w HMA
        if in_position and position_side < 0 and price_above_hma_1w:
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