#!/usr/bin/env python3
"""
Experiment #155: 12h RSI Mean Reversion + 1d HMA Trend Filter + ATR Stop

Hypothesis: 12h timeframe provides balance between trade frequency and signal quality.
RSI mean reversion works well in bear/range markets (2025 test period) while 1d HMA
provides trend bias to avoid counter-trend trades. This combines proven mean reversion
(RSI extremes) with trend filtering (HTF HMA) for asymmetric entries.

Why 12h might work:
- Slower than 15m/30m/1h that have been failing recently
- Fewer trades = less fee drag, higher quality signals
- RSI mean reversion has historical edge in crypto (75%+ win rate at extremes)
- 1d HMA trend filter avoids buying in bear markets / shorting in bull markets

Learning from failures:
- #145-154: All negative Sharpe - trend following doesn't work in 2025 bear market
- Need mean reversion component for range/bear conditions
- Simple conditions ensure adequate trade frequency (avoid 0-trade problem)

Timeframe: 12h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.35 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_rsi_meanrev_1d_hma_atr_v1"
timeframe = "12h"
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
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50.0).values
    return rsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper.values, lower.values, sma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.35
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 1d HMA = higher timeframe trend bias
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === RSI MEAN REVERSION SIGNALS ===
        # Oversold: RSI < 30 (potential long)
        # Overbought: RSI > 70 (potential short)
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        rsi_neutral = (rsi[i] >= 45) and (rsi[i] <= 55)
        
        # === BOLLINGER BAND CONFIRMATION ===
        # Price at lower band supports long
        # Price at upper band supports short
        at_bb_lower = close[i] <= bb_lower[i]
        at_bb_upper = close[i] >= bb_upper[i]
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        # Primary: RSI oversold + 1d bullish trend (pullback in uptrend)
        # Secondary: RSI very oversold (<20) regardless of trend (strong mean reversion)
        if rsi_oversold and bull_trend_1d:
            new_signal = SIZE_BASE
        elif rsi[i] < 20:  # Very oversold - strong mean reversion play
            new_signal = SIZE_STRONG
        elif rsi_oversold and at_bb_lower:  # RSI + BB confirmation
            new_signal = SIZE_BASE
        
        # === SHORT ENTRY CONDITIONS ===
        # Primary: RSI overbought + 1d bearish trend (rally in downtrend)
        # Secondary: RSI very overbought (>80) regardless of trend
        if rsi_overbought and bear_trend_1d:
            new_signal = -SIZE_BASE
        elif rsi[i] > 80:  # Very overbought - strong mean reversion play
            new_signal = -SIZE_STRONG
        elif rsi_overbought and at_bb_upper:  # RSI + BB confirmation
            new_signal = -SIZE_BASE
        
        # === EXIT CONDITIONS (RSI mean reversion) ===
        # Exit long when RSI crosses above 55 (mean reached)
        # Exit short when RSI crosses below 45 (mean reached)
        if in_position and position_side > 0 and rsi[i] > 55:
            new_signal = 0.0
        
        if in_position and position_side < 0 and rsi[i] < 45:
            new_signal = 0.0
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        # Update trailing highs/lows for active positions
        if in_position and position_side > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            # Trailing stop: 2.5 * ATR below highest close
            stoploss_price = highest_close - 2.5 * atr[i]
            if close[i] < stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        if in_position and position_side < 0:
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            # Trailing stop: 2.5 * ATR above lowest close
            stoploss_price = lowest_close + 2.5 * atr[i]
            if close[i] > stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        # Update position tracking
        # Entering new position
        if new_signal != 0.0 and not in_position:
            in_position = True
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Reversing position
        elif new_signal != 0.0 and in_position and np.sign(new_signal) != position_side:
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Exiting position
        elif new_signal == 0.0 and in_position:
            in_position = False
            position_side = 0
            entry_price = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals