#!/usr/bin/env python3
"""
Experiment #146: 12h Primary + 1d HTF — Donchian Breakout + HMA Trend + RSI Filter

Hypothesis: Recent failures show over-complex strategies generate 0 trades or negative Sharpe.
This strategy SIMPLIFIES entry conditions while keeping quality filters:

1. DONCHIAN(20) BREAKOUT: Clean breakout signal (price > 20-period high = long)
2. 1d HMA(21) SLOPE: Major trend bias (only long if 1d bullish, only short if 1d bearish)
3. RSI(14) FILTER: Avoid extreme entries (RSI 35-65 for breakouts, prevents chasing)
4. CHOPPINESS INDEX: Reduce size in choppy markets (CHOP>55 = 0.5x size)
5. ATR(14) TRAILING STOP: 2.5x ATR stoploss via signal→0

Why this should work:
- Donchian breakouts are proven trend-following signals (Turtle Trading)
- 1d HTF prevents counter-trend trades in strong moves
- RSI filter avoids buying tops/selling bottoms
- Simpler logic = more trades (target 30-50/year)
- 12h timeframe = low fee drag, proven to work in history

Timeframe: 12h (REQUIRED)
HTF: 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.30 base, 0.15 in choppy markets
Stoploss: 2.5 * ATR(14) trailing
Target trades: 30-50/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_hma_rsi_1d_v1"
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
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high, lowest low over period)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_hma_slope(hma_values, lookback=3):
    """Calculate HMA slope as percentage change over lookback periods."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        if hma_values[i - lookback] != 0:
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback] * 100
    return slope

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = range market (mean revert)
    CHOP < 38.2 = trend market (trend follow)
    """
    atr_values = calculate_atr(high, low, close, period)
    
    atr_sum = pd.Series(atr_values).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    price_range = np.where(price_range == 0, 1e-10, price_range)
    
    chop = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_slope = calculate_hma_slope(hma_1d_21, 3)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    chop_14 = calculate_choppiness(high, low, close, 14)
    
    # Breakout detection (price breaks Donchian)
    breakout_long = close > np.roll(donchian_upper, 1)
    breakout_short = close < np.roll(donchian_lower, 1)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.30
    CHOP_SIZE = 0.15
    
    # Track position state
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
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_slope_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]):
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === 1D TREND BIAS ===
        # Bullish: HMA slope positive and price above HMA
        trend_1d_bullish = hma_1d_slope_aligned[i] > 0.2
        # Bearish: HMA slope negative and price below HMA
        trend_1d_bearish = hma_1d_slope_aligned[i] < -0.2
        # Neutral: slope near zero
        trend_1d_neutral = not trend_1d_bullish and not trend_1d_bearish
        
        # === CHOPPINESS REGIME ===
        is_choppy = chop_14[i] > 55
        is_trending = chop_14[i] < 45
        
        # === POSITION SIZING ===
        current_size = CHOP_SIZE if is_choppy else BASE_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: Donchian breakout + 1d bullish bias + RSI filter
        if breakout_long[i]:
            # Must have bullish 1d trend OR neutral (don't fight strong bear)
            if trend_1d_bullish or trend_1d_neutral:
                # RSI filter: not overbought (avoid chasing)
                if 30 <= rsi_14[i] <= 70:
                    new_signal = current_size
                elif rsi_14[i] < 30 and trend_1d_bullish:
                    # Very oversold in bull trend = strong long
                    new_signal = current_size
        
        # SHORT ENTRY: Donchian breakout + 1d bearish bias + RSI filter
        if breakout_short[i]:
            # Must have bearish 1d trend OR neutral (don't fight strong bull)
            if trend_1d_bearish or trend_1d_neutral:
                # RSI filter: not oversold (avoid catching falling knife)
                if 30 <= rsi_14[i] <= 70:
                    new_signal = -current_size
                elif rsi_14[i] > 70 and trend_1d_bearish:
                    # Very overbought in bear trend = strong short
                    new_signal = -current_size
        
        # === FREQUENCY BOOSTER ===
        # If no trades for 100 bars (~50 days on 12h), loosen conditions
        if bars_since_last_trade > 100 and new_signal == 0.0 and not in_position:
            if trend_1d_bullish and rsi_14[i] < 45:
                new_signal = current_size * 0.5
            elif trend_1d_bearish and rsi_14[i] > 55:
                new_signal = -current_size * 0.5
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest price for long
                if close[i] > highest_price:
                    highest_price = close[i]
                # Trailing stop: highest - 2.5*ATR
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest price for short
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                # Trailing stop: lowest + 2.5*ATR
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        # Exit long if 1d trend turns bearish
        if in_position and position_side > 0 and trend_1d_bearish:
            stoploss_triggered = True
        
        # Exit short if 1d trend turns bullish
        if in_position and position_side < 0 and trend_1d_bullish:
            stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
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
            # If same direction, maintain position (no update needed)
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals