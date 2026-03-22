#!/usr/bin/env python3
"""
Experiment #579: 1h Volatility Spike Mean Reversion with 4h HMA Trend Bias

Hypothesis: After 500+ failed experiments, the pattern is clear:
1. Pure trend following fails on BTC/ETH (2022 crash + 2025 bear market)
2. Pure mean reversion fails without trend filter (catches falling knives)
3. VOLATILITY SPIKE + MEAN REVERSION + HTF TREND BIAS works best

Why this should work on 1h:
- ATR(7)/ATR(30) > 1.8 captures volatility spikes (panic/extremes)
- Price at BB(20, 2.0) extremes = oversold/overbought conditions
- 4h HMA trend bias prevents counter-trend entries (major failure mode)
- Asymmetric sizing: larger positions when HTF trend agrees with mean reversion
- 2*ATR stoploss protects against sustained moves (2022-style crashes)
- 1h timeframe = enough trades (20-50/year) without excessive fee drag

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 discrete (max 0.40)
Stoploss: 2.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_vol_spike_meanrev_4h_hma_bb_asymmetric_atr_v1"
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
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper.values, lower.values, sma.values

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, 20, 2.0)
    rsi_14 = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_REDUCED = 0.15
    
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
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(atr_7[i]) or np.isnan(atr_30[i]) or atr_30[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi_14[i]):
            signals[i] = 0.0
            continue
        
        # === 4H HMA TREND BIAS ===
        bull_bias = close[i] > hma_4h_aligned[i]
        bear_bias = close[i] < hma_4h_aligned[i]
        
        # === VOLATILITY SPIKE DETECTION ===
        # ATR ratio > 1.8 = volatility spike (panic/extreme conditions)
        atr_ratio = atr_7[i] / atr_30[i] if atr_30[i] > 0 else 0
        vol_spike = atr_ratio > 1.8
        
        # === BOLLINGER BAND EXTREMES ===
        # Price at lower band = oversold, at upper band = overbought
        bb_position = (close[i] - bb_lower[i]) / (bb_upper[i] - bb_lower[i]) if (bb_upper[i] - bb_lower[i]) > 0 else 0.5
        at_bb_lower = close[i] <= bb_lower[i] * 1.002  # Within 0.2% of lower band
        at_bb_upper = close[i] >= bb_upper[i] * 0.998  # Within 0.2% of upper band
        
        # === RSI CONFIRMATION ===
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        
        # === ENTRY LOGIC (Asymmetric based on HTF trend) ===
        new_signal = 0.0
        
        # Long entry: volatility spike + price at BB lower + RSI oversold
        # Larger size if 4h trend is bullish, smaller if bearish (counter-trend)
        if vol_spike and at_bb_lower and rsi_oversold:
            if bull_bias:
                new_signal = SIZE_BASE  # Trend-aligned mean reversion
            elif bear_bias:
                new_signal = SIZE_REDUCED  # Counter-trend (smaller size)
        
        # Short entry: volatility spike + price at BB upper + RSI overbought
        # Larger size if 4h trend is bearish, smaller if bullish (counter-trend)
        elif vol_spike and at_bb_upper and rsi_overbought:
            if bear_bias:
                new_signal = -SIZE_BASE  # Trend-aligned mean reversion
            elif bull_bias:
                new_signal = -SIZE_REDUCED  # Counter-trend (smaller size)
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.0 * atr_14[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.0 * atr_14[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit if 4h HMA flips strongly against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_bias:
                # Long position but 4h trend turned bearish
                # Only exit if price also dropped significantly
                if close[i] < entry_price * 0.97:  # 3% drop
                    new_signal = 0.0
            if position_side < 0 and bull_bias:
                # Short position but 4h trend turned bullish
                # Only exit if price also rose significantly
                if close[i] > entry_price * 1.03:  # 3% rise
                    new_signal = 0.0
        
        # === VOLATILITY NORMALIZATION EXIT ===
        # Exit when volatility returns to normal (ATR ratio < 1.2)
        if in_position and atr_ratio < 1.2:
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