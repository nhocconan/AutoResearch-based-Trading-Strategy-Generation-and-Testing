#!/usr/bin/env python3
"""
Experiment #464: 4h Primary + 12h/1d HTF — HMA Trend + RSI Pullback + Choppiness Sizing

Hypothesis: After analyzing 463 failed experiments, clear pattern emerges:
1. Simple HMA + RSI works (current best Sharpe=0.435 uses exactly this)
2. Too many filters = 0 trades (see exp 453, 455, 458 with Sharpe=0.000)
3. Choppiness should modulate SIZE not block entries (failed in exp 454, 461)
4. 4h timeframe naturally produces 20-50 trades/year (perfect for fee efficiency)
5. Asymmetric sizing (0.30 long, 0.25 short) protects in bear markets like 2022/2025

Why this might beat current best (Sharpe=0.435 on 1d):
- 4h captures more swings than 1d while avoiding 1h fee drag
- 12h HMA provides trend confirmation without excessive lag
- RSI(14) pullback entries are proven (current best uses RSI(14))
- Choppiness adjusts sizing: full size in trends, half in ranges
- ATR 2.5x trailing stop protects in 2022-style crashes
- Fewer conflicting conditions = more trades = better statistical significance

Position sizing: 0.25-0.30 (discrete levels, max 0.40)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: 30-50 trades/year on 4h, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_chop_size_12h1d_v1"
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
    """Calculate Hull Moving Average (HMA)."""
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        return series.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    hma_raw = 2.0 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_n)
    
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    CHOP > 61.8 = ranging market (reduce position size)
    CHOP < 38.2 = trending market (full position size)
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    atr = calculate_atr(high, low, close, period)
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    highest_high = high_s.rolling(window=period, min_periods=period).max().values
    lowest_low = low_s.rolling(window=period, min_periods=period).min().values
    
    range_hl = highest_high - lowest_low
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    
    chop = 100.0 * np.log10(atr_sum / range_hl) / np.log10(period)
    
    return chop

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators (major trend direction)
    hma_12h_21 = calculate_hma(df_12h['close'].values, period=21)
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    hma_4h_21 = calculate_hma(close, period=21)
    hma_4h_50 = calculate_hma(close, period=50)
    rsi_14 = calculate_rsi(close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    sma_200 = calculate_sma(close, period=200)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_LONG_SIZE = 0.30
    BASE_SHORT_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_price = 0.0
    lowest_price = 0.0
    entry_price = 0.0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_12h_21_aligned[i]) or np.isnan(hma_1d_21_aligned[i]):
            continue
        if np.isnan(hma_4h_21[i]) or np.isnan(hma_4h_50[i]):
            continue
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]) or np.isnan(sma_200[i]):
            continue
        
        # === HTF TREND DIRECTION (12h + 1d confluence) ===
        # Bullish: price above both 12h and 1d HMA
        # Bearish: price below both 12h and 1d HMA
        htf_bullish = (close[i] > hma_12h_21_aligned[i]) and (close[i] > hma_1d_21_aligned[i])
        htf_bearish = (close[i] < hma_12h_21_aligned[i]) and (close[i] < hma_1d_21_aligned[i])
        
        # === 4H LOCAL TREND (HMA crossover) ===
        hma_bullish = hma_4h_21[i] > hma_4h_50[i]
        hma_bearish = hma_4h_21[i] < hma_4h_50[i]
        
        # === SMA200 FILTER (long-term trend confirmation) ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === CHOPPINESS-BASED POSITION SIZING ===
        # Trending market (CHOP < 45): full size
        # Ranging market (CHOP > 55): half size
        # Neutral: 75% size
        if chop_14[i] < 45.0:
            size_multiplier = 1.0
        elif chop_14[i] > 55.0:
            size_multiplier = 0.5
        else:
            size_multiplier = 0.75
        
        # === RSI PULLBACK ENTRY CONDITIONS ===
        # Long: RSI pulls back to 35-45 in uptrend
        # Short: RSI rallies to 55-65 in downtrend
        rsi_long_pullback = 35.0 <= rsi_14[i] <= 50.0
        rsi_short_rally = 50.0 <= rsi_14[i] <= 65.0
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        
        # === ENTRY LOGIC (SIMPLE - FEWER CONFLICTING FILTERS) ===
        new_signal = 0.0
        
        # LONG ENTRIES - need HTF bullish OR (4h bullish + above SMA200)
        long_condition = htf_bullish or (hma_bullish and above_sma200)
        
        if long_condition:
            # Primary: RSI pullback in uptrend
            if rsi_long_pullback and hma_bullish:
                new_signal = BASE_LONG_SIZE * size_multiplier
            # Secondary: RSI oversold + any bullish signal
            elif rsi_oversold and (hma_bullish or above_sma200):
                new_signal = BASE_LONG_SIZE * size_multiplier * 0.8
            # Tertiary: HMA crossover confirmation
            elif hma_bullish and rsi_14[i] < 55.0 and not rsi_overbought:
                new_signal = BASE_LONG_SIZE * size_multiplier * 0.6
        
        # SHORT ENTRIES - need HTF bearish OR (4h bearish + below SMA200)
        short_condition = htf_bearish or (hma_bearish and below_sma200)
        
        if short_condition and new_signal == 0.0:
            # Primary: RSI rally in downtrend
            if rsi_short_rally and hma_bearish:
                new_signal = -BASE_SHORT_SIZE * size_multiplier
            # Secondary: RSI overbought + any bearish signal
            elif rsi_overbought and (hma_bearish or below_sma200):
                new_signal = -BASE_SHORT_SIZE * size_multiplier * 0.8
            # Tertiary: HMA crossover confirmation
            elif hma_bearish and rsi_14[i] > 45.0 and not rsi_oversold:
                new_signal = -BASE_SHORT_SIZE * size_multiplier * 0.6
        
        # === TRADE FREQUENCY BOOST (CRITICAL - avoid 0 trades) ===
        # If no position and weak signal, enter on simpler conditions
        if not in_position and new_signal == 0.0:
            # Long: RSI < 40 + price > 12h HMA (simpler entry)
            if rsi_14[i] < 40.0 and close[i] > hma_12h_21_aligned[i]:
                new_signal = BASE_LONG_SIZE * 0.5
            # Short: RSI > 60 + price < 12h HMA (simpler entry)
            elif rsi_14[i] > 60.0 and close[i] < hma_12h_21_aligned[i]:
                new_signal = -BASE_SHORT_SIZE * 0.5
        
        # === STOPLOSS CHECK (BEFORE exit logic - CRITICAL) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_price = max(highest_price, close[i])
            stop_price = highest_price - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_price == 0.0:
                lowest_price = close[i]
            else:
                lowest_price = min(lowest_price, close[i])
            stop_price = lowest_price + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT CONDITIONS ===
        # RSI extreme exit (take profit on exhaustion)
        if in_position and position_side > 0 and rsi_14[i] > 75.0:
            new_signal = 0.0
        if in_position and position_side < 0 and rsi_14[i] < 25.0:
            new_signal = 0.0
        
        # HTF trend reversal exit
        if in_position and position_side > 0 and htf_bearish:
            new_signal = 0.0
        if in_position and position_side < 0 and htf_bullish:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
        
        signals[i] = new_signal
    
    return signals