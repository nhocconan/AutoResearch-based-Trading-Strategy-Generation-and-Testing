#!/usr/bin/env python3
"""
Experiment #209: 12h Regime-Adaptive Strategy with 1d/1w HTF Context

Hypothesis: 12h timeframe benefits from regime detection rather than pure trend-following.
Previous 12h Donchian strategies failed (Sharpe=-0.192) because they traded breakouts
in ranging markets. This strategy uses Choppiness Index to detect regime:
- CHOP < 38.2 (trending): Donchian breakout with 1d HMA bias
- CHOP > 61.8 (ranging): RSI mean-reversion at Bollinger bands
- Middle zone: Stay flat or reduce position

Why this might work:
- Regime adaptation prevents whipsaw losses in ranges
- 1d HMA provides stable trend bias (proven in best strategies)
- 1w HMA adds ultra-HTF context for major trend direction
- 12h captures swing moves without 4h noise
- Discrete sizing (0.25) controls drawdown in crashes

Learning from failures:
- #197 (12h Donchian): -0.192 Sharpe - no regime filter
- #203 (12h CRSI+Chop): -0.351 Sharpe - too many conditions
- #202 (4h Asymmetric): -0.991 Sharpe - regime logic was flawed
- Need simpler regime detection + flexible entry conditions

Timeframe: 12h (REQUIRED)
HTF: 1d + 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_regime_adaptive_1d_1w_hma_chop_atr_v1"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    chop = np.zeros(n)
    
    for i in range(period, n):
        # Calculate ATR for each bar in the lookback
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr1 = high[j] - low[j]
            tr2 = abs(high[j] - close[j-1]) if j > 0 else tr1
            tr3 = abs(low[j] - close[j-1]) if j > 0 else tr1
            tr = max(tr1, tr2, tr3)
            atr_sum += tr
        
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 0 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0  # neutral
    
    # Fill initial values
    chop[:period] = 50.0
    
    return chop

def calculate_bollinger(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, lower, sma

def calculate_rsi(close, period=14):
    """Calculate RSI (Relative Strength Index)."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper/lower bounds)."""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(period-1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    # Fill initial values
    upper[:period-1] = upper[period-1]
    lower[:period-1] = lower[period-1]
    
    return upper, lower

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
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    chop = calculate_choppiness(high, low, close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, 20, 2.0)
    rsi = calculate_rsi(close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_REDUCED = 0.15
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]) or np.isnan(bb_upper[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 1d HMA = medium-term trend bias
        # 1w HMA = long-term trend bias
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION ===
        # CHOP < 38.2 = trending market (use breakout logic)
        # CHOP > 61.8 = ranging market (use mean-reversion logic)
        # 38.2 <= CHOP <= 61.8 = transition (reduce position or flat)
        is_trending = chop[i] < 38.2
        is_ranging = chop[i] > 61.8
        is_transition = not is_trending and not is_ranging
        
        new_signal = 0.0
        
        # === TRENDING REGIME: Donchian Breakout ===
        if is_trending:
            # Long: 1d bullish + breakout above Donchian
            if bull_trend_1d and close[i] > donchian_upper[i-1]:
                # Confirm with 1w trend if possible
                if bull_trend_1w or not np.isnan(hma_1w_aligned[i]):
                    new_signal = SIZE_BASE
            
            # Short: 1d bearish + breakout below Donchian
            if bear_trend_1d and close[i] < donchian_lower[i-1]:
                # Confirm with 1w trend if possible
                if bear_trend_1w or not np.isnan(hma_1w_aligned[i]):
                    new_signal = -SIZE_BASE
        
        # === RANGING REGIME: RSI Mean-Reversion at Bollinger Bands ===
        elif is_ranging:
            # Long: Price at BB lower + RSI oversold + 1w bullish bias preferred
            if close[i] <= bb_lower[i] and rsi[i] < 35:
                # Prefer long in bullish 1w trend, but allow in ranging
                if bull_trend_1w or is_ranging:
                    new_signal = SIZE_BASE
            
            # Short: Price at BB upper + RSI overbought + 1w bearish bias preferred
            if close[i] >= bb_upper[i] and rsi[i] > 65:
                # Prefer short in bearish 1w trend, but allow in ranging
                if bear_trend_1w or is_ranging:
                    new_signal = -SIZE_BASE
        
        # === TRANSITION REGIME: Reduce position or stay flat ===
        elif is_transition:
            # Only maintain existing positions, don't enter new ones
            # This reduces whipsaw in uncertain markets
            if in_position:
                new_signal = position_side * SIZE_REDUCED
            else:
                new_signal = 0.0
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        # Check stoploss on EXISTING position before considering new entry
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                # Trailing stop: 2.5 * ATR below highest close
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                # Trailing stop: 2.5 * ATR above lowest close
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
        
        # === UPDATE POSITION TRACKING FOR NEXT BAR ===
        if new_signal != 0.0:
            if not in_position:
                # Entering new position
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            # else: maintaining same position direction
        else:
            # Exiting position (signal-based or stoploss)
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals