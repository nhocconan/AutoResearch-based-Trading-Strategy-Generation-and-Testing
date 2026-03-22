#!/usr/bin/env python3
"""
Experiment #220: 4h Regime-Adaptive Strategy with 1d/1w HMA Bias + Choppiness Filter

Hypothesis: 4h timeframe with regime-adaptive logic should outperform pure trend or 
pure mean-reversion strategies. Using Choppiness Index (CHOP) to detect market state:
- CHOP > 61.8 = ranging market → mean reversion at RSI extremes
- CHOP < 38.2 = trending market → trend following with breakout
- 1d HMA provides higher timeframe bias (only trade with HTF trend)
- 1w HMA provides macro bias filter (avoid counter-macro trades)

Why this should work on 4h:
- 4h captures swing moves without 1h noise
- Regime detection prevents trend strategies in ranges (major failure mode)
- Mean reversion in ranges catches BTC/ETH oscillations
- HTF filters (1d/1w HMA) prevent counter-trend disasters like 2022
- Conservative sizing (0.25) controls drawdown

Learning from failures:
- #214 (4h DEMA): Sharpe=0.191 - worked but can improve with regime filter
- #208 (4h BB squeeze): Sharpe=-0.372 - BB alone insufficient
- #219 (1h Fisher): Sharpe=0.000 - 0 trades, conditions too strict
- Pure trend fails in ranges, pure mean-reversion fails in trends
- Need BOTH logics with regime switch

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d and 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_regime_chop_1d_1w_hma_rsi_atr_v1"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    chop = np.zeros(n)
    
    # Calculate ATR for each bar
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0  # neutral
    
    # Fill initial values
    chop[:period] = chop[period] if period < n else 50.0
    
    return chop

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    return sma.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper/lower bounds)."""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(period-1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    # Fill initial values
    upper[:period-1] = upper[period-1] if period-1 < n else high[0]
    lower[:period-1] = lower[period-1] if period-1 < n else low[0]
    
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
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    chop = calculate_choppiness(high, low, close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    sma_200 = calculate_sma(close, 200)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_HALF = 0.15
    
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
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]) or np.isnan(rsi[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME BIAS ===
        # 1d HMA = intermediate trend bias
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # 1w HMA = macro trend bias (stronger filter)
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION ===
        # CHOP > 61.8 = ranging market (mean reversion)
        # CHOP < 38.2 = trending market (trend following)
        # 38.2 - 61.8 = transition zone (use trend bias)
        is_range = chop[i] > 61.8
        is_trend = chop[i] < 38.2
        
        # === SMA200 FILTER ===
        # Only long if price > SMA200, only short if price < SMA200
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === EMA STRUCTURE ===
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        
        new_signal = 0.0
        
        # === REGIME 1: RANGING MARKET (Mean Reversion) ===
        if is_range:
            # Long: RSI < 30 (oversold) + 1d bullish bias + above SMA200
            if rsi[i] < 30 and bull_trend_1d and above_sma200:
                new_signal = SIZE_BASE
            
            # Short: RSI > 70 (overbought) + 1d bearish bias + below SMA200
            elif rsi[i] > 70 and bear_trend_1d and below_sma200:
                new_signal = -SIZE_BASE
        
        # === REGIME 2: TRENDING MARKET (Trend Following) ===
        elif is_trend:
            # Long: 1d bullish + 1w bullish + EMA bullish + Donchian breakout
            breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
            if bull_trend_1d and bull_trend_1w and ema_bullish and breakout_long:
                new_signal = SIZE_BASE
            
            # Short: 1d bearish + 1w bearish + EMA bearish + Donchian breakdown
            breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
            if bear_trend_1d and bear_trend_1w and ema_bearish and breakout_short:
                new_signal = -SIZE_BASE
        
        # === REGIME 3: TRANSITION ZONE (Use HTF bias only) ===
        else:
            # Only take trades aligned with BOTH 1d and 1w trend
            if bull_trend_1d and bull_trend_1w and ema_bullish and rsi[i] > 50:
                new_signal = SIZE_HALF
            elif bear_trend_1d and bear_trend_1w and ema_bearish and rsi[i] < 50:
                new_signal = -SIZE_HALF
        
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
            
            elif position_side < 0:
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