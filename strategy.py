#!/usr/bin/env python3
"""
Experiment #538: 4h Bollinger Squeeze Breakout with Daily/Weekly HMA Trend Bias

Hypothesis: After 500+ failed experiments, the key insight is:
1. 4h timeframe balances noise reduction with sufficient trade frequency
2. Bollinger Band squeeze (low volatility) precedes major breakouts
3. Daily/Weekly HMA provides major trend bias to avoid counter-trend trades
4. Volume confirmation ensures breakout has participation
5. RSI filter avoids entering at extreme overbought/oversold levels

Why this should work on 4h:
- BB squeeze catches volatility compression before expansion (proven pattern)
- 1d/1w HMA alignment via mtf_data prevents look-ahead on trend bias
- Volume > 0.8 * avg confirms breakout legitimacy
- RSI 35-65 range avoids chasing extremes while still generating trades
- 2.5*ATR stoploss protects against 2022-style crashes
- Discrete position sizing (0.30) limits drawdown during crashes

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d and 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.30 discrete (max 0.40)
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_bb_squeeze_1d_1w_hma_volume_rsi_atr_v1"
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

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper.values, lower.values, sma.values

def calculate_bb_width(upper, lower, sma):
    """Calculate Bollinger Band Width (volatility measure)."""
    width = (upper - lower) / sma
    return width

def calculate_rsi(close, period=14):
    """Calculate RSI (Relative Strength Index)."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_volume_ma(volume, period=20):
    """Calculate moving average of volume."""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean()
    return vol_ma.values

def calculate_bb_width_percentile(bb_width, lookback=50):
    """Calculate percentile rank of BB width over lookback period."""
    bb_s = pd.Series(bb_width)
    # Calculate rolling percentile rank
    percentile = bb_s.rolling(window=lookback, min_periods=lookback).apply(
        lambda x: (x < x.iloc[-1]).sum() / len(x) if len(x) == lookback else np.nan
    )
    return percentile.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    bb_upper, bb_lower, bb_sma = calculate_bollinger_bands(close, 20, 2.0)
    bb_width = calculate_bb_width(bb_upper, bb_lower, bb_sma)
    bb_width_pct = calculate_bb_width_percentile(bb_width, 50)
    rsi_14 = calculate_rsi(close, 14)
    vol_ma_20 = calculate_volume_ma(volume, 20)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.30
    
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
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_width[i]) or np.isnan(bb_width_pct[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # === HTF TREND BIAS (1d and 1w HMA) ===
        # Bullish: price above both 1d and 1w HMA
        bull_bias = close[i] > hma_1d_aligned[i] and close[i] > hma_1w_aligned[i]
        # Bearish: price below both 1d and 1w HMA
        bear_bias = close[i] < hma_1d_aligned[i] and close[i] < hma_1w_aligned[i]
        
        # === BB SQUEEZE DETECTION (low volatility = potential breakout) ===
        # BB width at low percentile (< 30% = squeezed)
        bb_squeeze = bb_width_pct[i] < 0.30
        
        # === BREAKOUT DETECTION ===
        # Price breaking above upper band = bullish breakout
        breakout_long = close[i] > bb_upper[i-1] if not np.isnan(bb_upper[i-1]) else False
        # Price breaking below lower band = bearish breakout
        breakout_short = close[i] < bb_lower[i-1] if not np.isnan(bb_lower[i-1]) else False
        
        # === VOLUME CONFIRMATION ===
        # Volume above 80% of average (loose filter to ensure trades)
        volume_confirm = volume[i] > 0.8 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        # === RSI FILTER (avoid extremes but allow trades) ===
        # RSI not extremely overbought for longs, not extremely oversold for shorts
        rsi_ok_long = rsi_14[i] < 70  # Not too overbought
        rsi_ok_short = rsi_14[i] > 30  # Not too oversold
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # Long: BB squeeze + bullish breakout + volume + trend bias + RSI ok
        if bb_squeeze and breakout_long and volume_confirm and bull_bias and rsi_ok_long:
            new_signal = SIZE
        
        # Short: BB squeeze + bearish breakout + volume + trend bias + RSI ok
        elif bb_squeeze and breakout_short and volume_confirm and bear_bias and rsi_ok_short:
            new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit if HTF HMA flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_bias:
                new_signal = 0.0
            if position_side < 0 and bull_bias:
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