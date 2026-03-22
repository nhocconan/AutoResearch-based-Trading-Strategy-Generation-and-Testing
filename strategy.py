#!/usr/bin/env python3
"""
Experiment #493: 15m Momentum Trend-Follow with 4h HMA Bias

Hypothesis: After analyzing 492 failed experiments, the pattern is clear:
1. Mean-reversion FAILS on 15m (too much noise, Sharpe=-14.3 on exp#481)
2. Complex regime filters (CHOP, ADX>25) create TOO FEW trades
3. RSI extremes (30/70) are too strict for consistent entries

This strategy uses SIMPLE momentum trend-following:
1. 4h HMA(21) for trend bias (via mtf_data helper - call ONCE)
2. 15m ROC(10) for momentum confirmation
3. RSI(14) with LOOSE thresholds (25-75) for entry timing
4. Volume > SMA(20) for confirmation (not too strict)
5. ATR(14) 2.0x trailing stoploss

Key changes from failures:
- LOOSE RSI: 25-75 instead of 30-70 or 35-65 (more trades)
- NO ADX filter (ADX>25 kills trade count)
- NO choppiness index (exp#483 failed with it)
- Simple trend + momentum = more reliable on 15m

Timeframe: 15m (REQUIRED for exp#493)
HTF: 4h via mtf_data.get_htf_data (ONCE before loop)
Position sizing: 0.25 discrete (conservative for 15m volatility)
Stoploss: 2.0 * ATR(14) trailing
Target: 40-80 trades/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_momentum_4h_hma_roc_rsi_vol_atr_v1"
timeframe = "15m"
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

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    return close_s.ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_roc(close, period=10):
    """Calculate Rate of Change (momentum)."""
    close_s = pd.Series(close)
    roc = close_s.pct_change(periods=period) * 100
    return roc.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h HMA trend
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    ema_21 = calculate_ema(close, 21)
    rsi = calculate_rsi(close, 14)
    roc = calculate_roc(close, 10)
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(roc[i]) or np.isnan(ema_21[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(vol_sma[i]) or vol_sma[i] == 0:
            signals[i] = 0.0
            continue
        
        # === 4H HMA TREND BIAS ===
        bull_trend = close[i] > hma_4h_aligned[i]
        bear_trend = close[i] < hma_4h_aligned[i]
        
        # === VOLUME CONFIRMATION (loose) ===
        vol_ok = volume[i] > 0.8 * vol_sma[i]  # Just above average, not strict
        
        # === MOMENTUM + RSI ENTRY (LOOSE thresholds for more trades) ===
        new_signal = 0.0
        
        # BULL TREND: Long on momentum + RSI confirmation
        if bull_trend:
            # Momentum positive (ROC > 0)
            if roc[i] > 0.5:
                # RSI not overbought (loose: < 75 instead of < 70)
                if rsi[i] < 75:
                    # Price above EMA (trend confirmation)
                    if close[i] > ema_21[i]:
                        new_signal = SIZE
        
        # BEAR TREND: Short on momentum + RSI confirmation
        elif bear_trend:
            # Momentum negative (ROC < 0)
            if roc[i] < -0.5:
                # RSI not oversold (loose: > 25 instead of > 30)
                if rsi[i] > 25:
                    # Price below EMA (trend confirmation)
                    if close[i] < ema_21[i]:
                        new_signal = -SIZE
        
        # === STOPLOSS: 2.0 * ATR trailing (Rule 6) ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.0 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.0 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit if 4h trend flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend:
                new_signal = 0.0
            if position_side < 0 and bull_trend:
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
                # Position flip
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