#!/usr/bin/env python3
"""
Experiment #012: 12h HMA-RSI Pullback with 1d/1w Trend Alignment

Hypothesis: Previous strategies failed due to:
- Choppiness Index regime switching (failed #001, #002, #003, #011)
- Over-complicated multi-regime logic (failed #007, #011)
- Too many conflicting filters causing 0 trades (failed #008, #010)

This strategy uses a PROVEN pattern from research notes:
"HMA crossover + RSI filter + ATR trail (SOL +0.879)"

Key innovations for 12h timeframe:
1. HMA(16/48) crossover - smoother than EMA, less lag than SMA
2. RSI(7) pullback entries - enter on dips in uptrend, rallies in downtrend
3. 1d HMA for intermediate trend bias - align with daily direction
4. 1w HMA for major trend filter - only trade with weekly trend
5. ATR(14) trailing stoploss - 2.5x ATR to protect capital
6. Volume confirmation - require 1.2x average volume on entry

Why 12h works:
- 20-50 trades/year target (manageable fee drag ~1-2.5%)
- Less noise than 4h, more signals than 1d
- Captures multi-day swings without whipsaw

Position sizing: 0.25-0.30 discrete levels (CRITICAL for drawdown control)
Stoploss: 2.5 * ATR(14) trailing
Timeframe: 12h (REQUIRED for this experiment)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_rsi_pullback_1d_1w_trend_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period):
    """
    Hull Moving Average - reduces lag while maintaining smoothness.
    Formula: HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    Reference: Alan Hull, 2005
    """
    close_s = pd.Series(close)
    n = len(close)
    
    if period < 2:
        return np.full(n, np.nan)
    
    # WMA helper
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, period)
    
    # HMA calculation
    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_n)
    
    return hma.values

def calculate_rsi(close, period=14):
    """
    Relative Strength Index - momentum oscillator.
    Reference: J. Welles Wilder, 1978
    """
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rs = rs.replace([np.inf, -np.inf], np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1D HMA for intermediate trend
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 1W HMA for major trend bias
    hma_1w_12 = calculate_hma(df_1w['close'].values, 12)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_12)
    
    # Calculate 12h indicators
    hma_12h_fast = calculate_hma(close, 16)
    hma_12h_slow = calculate_hma(close, 48)
    rsi_7 = calculate_rsi(close, 7)
    rsi_14 = calculate_rsi(close, 14)
    atr_14 = calculate_atr(high, low, close, 14)
    
    # Volume moving average for confirmation
    volume_s = pd.Series(volume)
    volume_ma20 = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        
        if np.isnan(hma_12h_fast[i]) or np.isnan(hma_12h_slow[i]):
            continue
        
        if np.isnan(rsi_7[i]) or np.isnan(rsi_14[i]):
            continue
        
        if np.isnan(volume_ma20[i]) or volume_ma20[i] == 0:
            continue
        
        # === 1W MAJOR TREND BIAS ===
        weekly_bullish = close[i] > hma_1w_aligned[i]
        weekly_bearish = close[i] < hma_1w_aligned[i]
        
        # === 1D INTERMEDIATE TREND ===
        daily_bullish = close[i] > hma_1d_aligned[i]
        daily_bearish = close[i] < hma_1d_aligned[i]
        
        # === 12H HMA TREND ===
        hma_bullish = hma_12h_fast[i] > hma_12h_slow[i]
        hma_bearish = hma_12h_fast[i] < hma_12h_slow[i]
        
        # === HMA SLOPE CONFIRMATION ===
        hma_slope_long = hma_12h_fast[i] > hma_12h_fast[i-3] if i > 3 else False
        hma_slope_short = hma_12h_fast[i] < hma_12h_fast[i-3] if i > 3 else False
        
        # === RSI PULLBACK CONDITIONS ===
        # Long: RSI dipped but still bullish zone (40-55)
        rsi_pullback_long = (rsi_7[i] > 35) & (rsi_7[i] < 55)
        # Short: RSI rallied but still bearish zone (45-60)
        rsi_pullback_short = (rsi_7[i] > 45) & (rsi_7[i] < 65)
        
        # RSI extremes for strong momentum
        rsi_strong_long = rsi_7[i] > 55
        rsi_strong_short = rsi_7[i] < 45
        
        # === VOLUME CONFIRMATION ===
        volume_ok = volume[i] > 1.15 * volume_ma20[i]  # 15% above average
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: Need 12h HMA bullish + RSI pullback + (1d/1w alignment OR volume)
        long_score = 0
        if hma_bullish:
            long_score += 2  # Primary requirement
        if hma_slope_long:
            long_score += 1
        if rsi_pullback_long or rsi_strong_long:
            long_score += 1
        if weekly_bullish:
            long_score += 1  # Major trend alignment
        if daily_bullish:
            long_score += 0.5  # Intermediate trend
        if volume_ok:
            long_score += 0.5
        
        # Enter long if score >= 4 (need trend + confirmation)
        if long_score >= 4 and hma_bullish:
            new_signal = BASE_SIZE
        
        # SHORT ENTRY: Need 12h HMA bearish + RSI pullback + (1d/1w alignment OR volume)
        short_score = 0
        if hma_bearish:
            short_score += 2  # Primary requirement
        if hma_slope_short:
            short_score += 1
        if rsi_pullback_short or rsi_strong_short:
            short_score += 1
        if weekly_bearish:
            short_score += 1  # Major trend alignment
        if daily_bearish:
            short_score += 0.5  # Intermediate trend
        if volume_ok:
            short_score += 0.5
        
        # Enter short if score >= 4 (need trend + confirmation)
        if short_score >= 4 and hma_bearish:
            new_signal = -BASE_SIZE
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 60 bars (~30 days on 12h), allow weaker entry
        if bars_since_last_trade > 60 and new_signal == 0.0 and not in_position:
            if hma_bullish and (daily_bullish or weekly_bullish):
                new_signal = BASE_SIZE * 0.6  # Smaller size
            elif hma_bearish and (daily_bearish or weekly_bearish):
                new_signal = -BASE_SIZE * 0.6
        
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
            # Exit long if 12h HMA turns bearish
            if position_side > 0 and hma_bearish:
                trend_reversal = True
            # Exit short if 12h HMA turns bullish
            if position_side < 0 and hma_bullish:
                trend_reversal = True
        
        # === RSI EXTREME EXIT ===
        rsi_exit = False
        if in_position and position_side != 0:
            # Exit long if RSI extremely overbought (>80)
            if position_side > 0 and rsi_7[i] > 80:
                rsi_exit = True
            # Exit short if RSI extremely oversold (<20)
            if position_side < 0 and rsi_7[i] < 20:
                rsi_exit = True
        
        # Apply stoploss or exits
        if stoploss_triggered or trend_reversal or rsi_exit:
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
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                # Exit position
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals