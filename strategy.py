#!/usr/bin/env python3
"""
Experiment #370: 4h Regime-Adaptive with 1d/1w HMA Bias + Volatility Filter

Hypothesis: After 369 failed experiments, the clearest pattern is that static strategies
fail because they don't adapt to market regime. For 4h timeframe specifically:

1. BOLLINGER BAND WIDTH REGIME: Detect trending vs ranging markets
   - BB Width < 30th percentile = trending (use breakout logic)
   - BB Width > 70th percentile = ranging (use mean reversion logic)
   - This is the #1 meta-filter from quantitative literature

2. 1d HMA TREND BIAS: Only take trades in direction of daily trend
   - Long only if price > 1d HMA(21)
   - Short only if price < 1d HMA(21)
   - Filters 50%+ of losing trades

3. 1w HMA MACRO BIAS: Additional filter for major trend direction
   - Adds conviction when 1d and 1w agree
   - Increases position size when aligned (0.30 vs 0.20)

4. DUAL ENTRY LOGIC:
   - Trending regime: Donchian(20) breakout in trend direction
   - Ranging regime: RSI(14) mean reversion (RSI<35 long, RSI>65 short)
   - Loosened thresholds to ensure sufficient trades

5. ATR TRAILING STOP (2.5x): Protect capital on reversals
   - Signal → 0 when price moves 2.5*ATR against position
   - Critical for limiting drawdown

6. POSITION SIZING: 0.20-0.30 discrete (conservative for 4h volatility)
   - 0.20 when 1d/1w disagree
   - 0.30 when 1d/1w agree (higher conviction)
   - Discrete levels minimize fee churn

Why 4h should work:
- Fast enough to capture swings, slow enough to filter noise
- 4h candles are significant for crypto (6 per day)
- Should generate 30-60 trades/year per symbol
- Regime adaptation handles both 2022 crash and 2025 bear market

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d and 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_regime_adaptive_1d_1w_hma_bb_rsi_donchian_atr_v1"
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
    """Calculate Bollinger Bands and Band Width."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    # Band width as percentage of price
    bw = (upper - lower) / sma * 100
    return upper, lower, bw

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_donchian_channels(high, low, period=20):
    """Calculate Donchian Channels using rolling max/min."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    upper = high_s.rolling(window=period, min_periods=period).max().values
    lower = low_s.rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_bb_width_percentile(bb_width, lookback=100):
    """Calculate rolling percentile of BB Width for regime detection."""
    bb_s = pd.Series(bb_width)
    # Calculate percentile rank over lookback period
    percentile = bb_s.rolling(window=lookback, min_periods=lookback).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) * 100 if x.max() > x.min() else 50
    )
    return percentile.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
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
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_width = calculate_bollinger_bands(close, 20, 2.0)
    bb_percentile = calculate_bb_width_percentile(bb_width, 100)
    donchian_upper, donchian_lower = calculate_donchian_channels(high, low, 20)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_LOW = 0.20   # When 1d/1w disagree
    SIZE_HIGH = 0.30  # When 1d/1w agree (higher conviction)
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(bb_percentile[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        # === HTF TREND BIAS ===
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION ===
        # BB Width percentile < 30 = trending regime (breakout logic)
        # BB Width percentile > 70 = ranging regime (mean reversion logic)
        trending_regime = bb_percentile[i] < 30
        ranging_regime = bb_percentile[i] > 70
        
        # === POSITION SIZING ===
        # Higher size when 1d and 1w agree on trend direction
        if bull_trend_1d and bull_trend_1w:
            size_long = SIZE_HIGH
        elif bear_trend_1d and bear_trend_1w:
            size_short = SIZE_HIGH
        else:
            size_long = SIZE_LOW
            size_short = SIZE_LOW
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # TRENDING REGIME: Donchian breakout in trend direction
        if trending_regime:
            # Long breakout + 1d bullish bias
            long_breakout = close[i] > donchian_upper[i-1] if i > 0 else False
            if long_breakout and bull_trend_1d:
                new_signal = size_long
            
            # Short breakout + 1d bearish bias
            short_breakout = close[i] < donchian_lower[i-1] if i > 0 else False
            if short_breakout and bear_trend_1d:
                new_signal = -size_short
        
        # RANGING REGIME: RSI mean reversion
        elif ranging_regime:
            # RSI oversold + 1d bullish bias = long
            if rsi[i] < 35 and bull_trend_1d:
                new_signal = size_long
            
            # RSI overbought + 1d bearish bias = short
            if rsi[i] > 65 and bear_trend_1d:
                new_signal = -size_short
        
        # NEUTRAL REGIME: Stay flat or hold existing position
        else:
            # In neutral regime, only exit signals, no new entries
            if in_position:
                new_signal = signals[i-1] if i > 0 else 0.0
            else:
                new_signal = 0.0
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit if 1d trend flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_1d:
                new_signal = 0.0
            if position_side < 0 and bull_trend_1d:
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