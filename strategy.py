#!/usr/bin/env python3
"""
Experiment #022: 12h Dual Regime Strategy with 1D/1W HMA Trend Filter

Hypothesis: After 21 failed experiments, the clearest pattern is:
1. 12h timeframe produces optimal trade frequency (20-50/year)
2. Pure trend strategies fail in bear/range markets (2022, 2025)
3. Mean reversion works in choppy markets, trend-follow in trending markets
4. Simpler entry conditions = more trades = better statistical significance

This strategy combines:

1. 1D HMA trend bias: Primary trend filter. Long only if price>1d_HMA,
   short only if price<1d_HMA. More responsive than 1W for 12h TF.

2. 1W HMA super-trend: Ultra-stable confirmation. Only take signals aligned
   with weekly trend direction. Prevents counter-trend trades.

3. Choppiness Index regime: CHOP>55 = range (mean revert), CHOP<45 = trend.
   Switches between mean reversion and trend-following logic.

4. RSI(14) extremes: Long when RSI<35, Short when RSI>65.
   Less extreme than CRSI<15/85 to generate more trades.

5. ATR-based stoploss: 2.5*ATR trailing stop to limit drawdown.

6. Simplified entry logic: Fewer confluence requirements = more trades.

Why this should work:
- 12h TF = 20-50 trades/year (optimal for fee drag vs signal quality)
- Dual regime adapts to market conditions (chop vs trend)
- 1D+1W HMA provides stable trend bias without whipsaw
- RSI(14) extremes more common than CRSI extremes = more trades
- Target: 30-50 trades/year, Sharpe > 0.5, DD < -30%

Timeframe: 12h (REQUIRED)
HTF: 1d and 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_dual_regime_1d_1w_hma_rsi_chop_atr_v1"
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
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 55 = range/choppy market (mean revert)
    CHOP < 45 = trending market (trend follow)
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr_sum = tr.rolling(window=period, min_periods=period).sum()
    hh_ll = high_s.rolling(window=period, min_periods=period).max() - \
            low_s.rolling(window=period, min_periods=period).min()
    
    chop = 100 * np.log10(atr_sum / hh_ll.replace(0, np.inf)) / np.log10(period)
    chop = chop.clip(0, 100).fillna(50).values
    
    return chop

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
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    chop = calculate_choppiness(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    SIZE_LONG = 0.30
    SIZE_SHORT = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = np.inf
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # === TREND BIAS (1D + 1W HMA) ===
        # Bullish: price above both 1D and 1W HMA
        # Bearish: price below both 1D and 1W HMA
        # Neutral: mixed signals (reduce position or skip)
        bull_1d = close[i] > hma_1d_aligned[i]
        bull_1w = close[i] > hma_1w_aligned[i]
        bear_1d = close[i] < hma_1d_aligned[i]
        bear_1w = close[i] < hma_1w_aligned[i]
        
        strong_bull = bull_1d and bull_1w
        strong_bear = bear_1d and bear_1w
        
        # === CHOPPINESS REGIME ===
        is_choppy = chop[i] > 55  # Range market
        is_trending = chop[i] < 45  # Trending market
        
        # === RSI SIGNALS ===
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # MODE 1: MEAN REVERSION IN CHOPPY MARKET
        if is_choppy:
            # Long: oversold RSI + neutral-to-bullish trend
            if rsi_oversold and (bull_1d or not bear_1w):
                new_signal = SIZE_LONG
            
            # Short: overbought RSI + neutral-to-bearish trend
            elif rsi_overbought and (bear_1d or not bull_1w):
                new_signal = -SIZE_SHORT
        
        # MODE 2: TREND FOLLOWING IN TRENDING MARKET
        elif is_trending:
            # Long: strong bullish trend + RSI not overbought
            if strong_bull and not rsi_overbought:
                new_signal = SIZE_LONG
            
            # Short: strong bearish trend + RSI not oversold
            elif strong_bear and not rsi_oversold:
                new_signal = -SIZE_SHORT
        
        # MODE 3: NEUTRAL REGIME (45-55 CHOP) - use trend bias only
        else:
            if strong_bull:
                new_signal = SIZE_LONG * 0.5  # Reduced size in uncertain regime
            elif strong_bear:
                new_signal = -SIZE_SHORT * 0.5
        
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
                if close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
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
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else np.inf
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else np.inf
            # If same side, keep tracking highest/lowest
            elif position_side > 0 and close[i] > highest_price:
                highest_price = close[i]
            elif position_side < 0 and close[i] < lowest_price:
                lowest_price = close[i]
        else:
            if in_position:
                # Exit position
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = np.inf
        
        signals[i] = new_signal
    
    return signals