#!/usr/bin/env python3
"""
Experiment #011: 4h HMA Trend + 1D HMA Bias + RSI Pullback with Z-Score Filter

Hypothesis: After 10 failed experiments with complex regime-switching logic, 
simplicity wins. The best proven strategy (mtf_hma_rsi_zscore_v1, Sharpe=5.4) 
used: HTF HMA trend + LTF RSI pullback + Z-score filter.

This strategy adapts that proven formula for 4h primary timeframe:
1. 1D HMA(21) = Primary trend bias (very stable, few whipsaws)
2. 4h RSI(14) pullback = Entry timing (RSI 30-55 for long, 45-70 for short)
3. Z-score(20) filter = Avoid extremes (|z| < 2.0 for entry)
4. ATR(14) trailing stop = Risk management (2.5 * ATR)
5. Discrete sizing = 0.25-0.30, ATR-scaled

Why this should work when others failed:
- Simpler logic = less overfitting to specific regimes
- 1D HMA = more stable than 4h HMA for trend direction
- RSI pullback (not extreme) = catches trend continuations, not reversals
- Z-score filter = avoids chasing moves
- Proven formula from best historical strategy
- Looser RSI thresholds = ensures ≥10 trades per symbol

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1D via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
Target trades: 20-50 per year (loose enough RSI thresholds)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_1d_hma_zscore_atr_v1"
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
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

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
    
    return rsi.values

def calculate_zscore(close, period=20):
    """Calculate Z-score of price relative to rolling mean."""
    close_s = pd.Series(close)
    mean = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    zscore = (close_s - mean) / std.replace(0, np.inf)
    
    return zscore.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    zscore_20 = calculate_zscore(close, 20)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]):
            continue
        
        if np.isnan(zscore_20[i]):
            continue
        
        # === 1D HMA TREND BIAS ===
        bull_bias = close[i] > hma_1d_aligned[i]
        bear_bias = close[i] < hma_1d_aligned[i]
        
        # === Z-SCORE FILTER (avoid extremes) ===
        zscore_ok = abs(zscore_20[i]) < 2.0
        
        # === ATR-BASED POSITION SIZING ===
        if i > 100:
            atr_median = np.nanmedian(atr_14[100:i])
            atr_ratio = atr_14[i] / atr_median if atr_median > 0 else 1.0
            atr_ratio = np.clip(atr_ratio, 0.5, 2.0)
            size_multiplier = 1.0 / atr_ratio
            current_size = BASE_SIZE * size_multiplier
            current_size = np.clip(current_size, 0.20, 0.35)
        else:
            current_size = BASE_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG: 1D bullish bias + RSI pullback (30-55) + z-score OK
        # Looser thresholds to ensure enough trades
        if bull_bias and zscore_ok:
            if 30 <= rsi_14[i] <= 55:
                new_signal = current_size
        
        # SHORT: 1D bearish bias + RSI pullback (45-70) + z-score OK
        # Looser thresholds to ensure enough trades
        if bear_bias and zscore_ok:
            if 45 <= rsi_14[i] <= 70:
                new_signal = -current_size
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest price for long position
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest price for short position
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if 1D bias turns bearish
            if position_side > 0 and bear_bias:
                trend_reversal = True
            # Exit short if 1D bias turns bullish
            if position_side < 0 and bull_bias:
                trend_reversal = True
        
        # Apply stoploss or trend reversal
        if stoploss_triggered or trend_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                # New entry
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                # Exit position
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
        
        signals[i] = new_signal
    
    return signals