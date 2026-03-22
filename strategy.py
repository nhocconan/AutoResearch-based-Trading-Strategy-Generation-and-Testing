#!/usr/bin/env python3
"""
Experiment #603: 1h Mean Reversion with 4h Trend Filter + Volatility Regime

Hypothesis: After analyzing 534+ failures, the key insight is that 1h timeframe
needs a balance between mean reversion (for range markets) and trend following
(for directional moves). Most failed strategies were either too strict (0 trades)
or too simple (EMA crossover always fails on BTC/ETH).

This strategy combines:
1. 4h HMA for trend bias (HTF via mtf_data - call ONCE before loop)
2. 1h Bollinger Bands for mean reversion entries
3. 1h RSI for momentum exhaustion confirmation
4. ATR-based stoploss at 2.0x
5. Volatility regime filter (BB Width percentile) to adjust entry thresholds

Why this should work on 1h:
- 4h trend filter prevents counter-trend mean reversion in strong trends
- BB + RSI combo generates frequent signals in range markets (2022, 2025)
- Looser RSI thresholds (30/70 vs 20/80) ensure trades trigger
- Discrete position sizing (0.25) controls drawdown during 2022 crash

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h via mtf_data.get_htf_data() - called ONCE before loop
Position sizing: 0.25 discrete (max 0.40)
Stoploss: 2.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_meanrev_4h_hma_bb_rsi_volregime_atr_v1"
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

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    width = (upper - lower) / sma
    
    return upper.values, lower.values, sma.values, width.values

def calculate_bb_width_percentile(bb_width, lookback=100):
    """Calculate percentile rank of BB Width over lookback period."""
    bb_width_s = pd.Series(bb_width)
    percentile = bb_width_s.rolling(window=lookback, min_periods=lookback).apply(
        lambda x: (x < x.iloc[-1]).sum() / len(x), raw=False
    )
    return percentile.values

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
    rsi_14 = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_mid, bb_width = calculate_bollinger(close, 20, 2.0)
    bb_width_pct = calculate_bb_width_percentile(bb_width, 100)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        
        if np.isnan(bb_width_pct[i]):
            continue
        
        # === 4H HMA TREND BIAS ===
        bull_bias = close[i] > hma_4h_aligned[i]
        bear_bias = close[i] < hma_4h_aligned[i]
        
        # === VOLATILITY REGIME (BB Width Percentile) ===
        low_vol = bb_width_pct[i] < 0.30  # BB width in bottom 30%
        high_vol = bb_width_pct[i] > 0.70  # BB width in top 30%
        
        # === BOLLINGER BAND EXTREMES ===
        # Price below lower band (oversold)
        below_bb_lower = close[i] < bb_lower[i]
        # Price above upper band (overbought)
        above_bb_upper = close[i] > bb_upper[i]
        # Price near lower band (within 1%)
        near_bb_lower = close[i] < bb_lower[i] * 1.01
        # Price near upper band (within 1%)
        near_bb_upper = close[i] > bb_upper[i] * 0.99
        
        # === RSI EXTREMES (loose thresholds for more trades) ===
        rsi_oversold = rsi_14[i] < 40  # Looser than traditional 30
        rsi_overbought = rsi_14[i] > 60  # Looser than traditional 70
        rsi_extreme_oversold = rsi_14[i] < 30
        rsi_extreme_overbought = rsi_14[i] > 70
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # MODE 1: LOW VOLATILITY (Range market) - Mean Reversion
        if low_vol:
            # Long: Price at/near BB lower + RSI oversold + 4h not strongly bearish
            if (below_bb_lower or near_bb_lower) and rsi_oversold:
                if not (bear_bias and rsi_14[i] < 35):  # Allow long even in bear if RSI very low
                    new_signal = SIZE
            
            # Short: Price at/near BB upper + RSI overbought + 4h not strongly bullish
            elif (above_bb_upper or near_bb_upper) and rsi_overbought:
                if not (bull_bias and rsi_14[i] > 65):  # Allow short even in bull if RSI very high
                    new_signal = -SIZE
        
        # MODE 2: HIGH VOLATILITY (Trending market) - Trend Following with Pullback
        elif high_vol:
            # Long: Bullish 4h bias + RSI pullback (not extreme)
            if bull_bias and 35 < rsi_14[i] < 55:
                # Enter on pullback in uptrend
                if close[i] > bb_mid[i]:
                    new_signal = SIZE
            
            # Short: Bearish 4h bias + RSI pullback (not extreme)
            elif bear_bias and 45 < rsi_14[i] < 65:
                # Enter on pullback in downtrend
                if close[i] < bb_mid[i]:
                    new_signal = -SIZE
        
        # MODE 3: NORMAL VOLATILITY - Standard Mean Reversion
        else:
            # Long: BB lower + RSI oversold
            if near_bb_lower and rsi_oversold:
                new_signal = SIZE
            
            # Short: BB upper + RSI overbought
            elif near_bb_upper and rsi_overbought:
                new_signal = -SIZE
        
        # === EXTREME RSI OVERRIDE (always take the trade) ===
        # This ensures we generate trades even when other conditions fail
        if rsi_extreme_oversold and not in_position:
            new_signal = SIZE
        elif rsi_extreme_overbought and not in_position:
            new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.0 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.0 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # Apply stoploss
        if stoploss_triggered:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                # New entry
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
                # Exit position
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals