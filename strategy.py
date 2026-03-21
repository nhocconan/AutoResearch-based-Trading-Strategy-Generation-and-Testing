#!/usr/bin/env python3
"""
Experiment #186: 1d HMA Trend + RSI Pullback with Weekly Filter
Hypothesis: Daily timeframe captures major crypto cycles while avoiding noise.
Using HMA(21/50) crossover for trend direction, RSI(14) pullbacks for entries.
Weekly HMA(21) provides macro bias filter. Choppiness Index switches between
trend-following (CHOP<45) and mean-reversion (CHOP>55) modes. Loosened RSI
thresholds (35/65) ensure sufficient trades on slow 1d timeframe. ATR stoploss
at 2.5*ATR limits drawdown during 2022-style crashes. Position sizing 0.30
balances return vs 77% BTC crash risk.

Key changes from failed experiments:
- LOOSER RSI thresholds (35/65 vs 30/70) to generate more trades
- Simpler logic - fewer conflicting filters that cause 0 trades
- Weekly HMA as single HTF reference (not daily+weekly which conflicted)
- Discrete signal levels (0.0, ±0.15, ±0.30) to minimize fee churn
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_hma_rsi_weekly_chop_atr_v1"
timeframe = "1d"
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
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate RSI indicator."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market (mean reversion)
    CHOP < 38.2 = trending market (trend following)
    """
    atr = calculate_atr(high, low, close, period)
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    range_hl = highest_high - lowest_low
    range_hl = np.where(range_hl > 0, range_hl, 1e-10)
    
    chop = 100 * np.log10(np.sum(atr) / (range_hl * period))
    chop = np.where(np.isnan(chop), 50.0, chop)
    chop = np.clip(chop, 0, 100)
    
    return chop

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    chop = calculate_choppiness(high, low, close, 14)
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.30
    SIZE_HALF = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(200, n):
        # Weekly trend filter (macro bias)
        weekly_bullish = hma_1w_aligned[i] > 0 and close[i] > hma_1w_aligned[i]
        weekly_bearish = hma_1w_aligned[i] > 0 and close[i] < hma_1w_aligned[i]
        
        # Regime detection
        is_ranging = chop[i] > 50.0
        is_trending = chop[i] < 45.0
        
        # Daily trend
        trend_bullish = hma_21[i] > hma_50[i] and close[i] > sma_200[i]
        trend_bearish = hma_21[i] < hma_50[i] and close[i] < sma_200[i]
        
        # HMA crossover signals
        hma_cross_bullish = hma_21[i] > hma_50[i] and hma_21[i-1] <= hma_50[i-1]
        hma_cross_bearish = hma_21[i] < hma_50[i] and hma_21[i-1] >= hma_50[i-1]
        
        # RSI signals (LOOSENED for more trades)
        rsi_oversold = rsi[i] < 40
        rsi_overbought = rsi[i] > 60
        rsi_rising = rsi[i] > rsi[i-1] if i > 0 else False
        rsi_falling = rsi[i] < rsi[i-1] if i > 0 else False
        rsi_neutral = 35 < rsi[i] < 65
        
        new_signal = 0.0
        
        # === TREND FOLLOWING MODE ===
        if is_trending:
            # Long: HMA crossover + RSI not overbought + weekly bullish
            if hma_cross_bullish and not rsi_overbought:
                if weekly_bullish or trend_bullish:
                    new_signal = SIZE_ENTRY
            
            # Short: HMA crossover + RSI not oversold + weekly bearish
            elif hma_cross_bearish and not rsi_oversold:
                if weekly_bearish or trend_bearish:
                    new_signal = -SIZE_ENTRY
            
            # Pullback entry in established trend
            elif trend_bullish and rsi_oversold and rsi_rising:
                if weekly_bullish:
                    new_signal = SIZE_ENTRY
            elif trend_bearish and rsi_overbought and rsi_falling:
                if weekly_bearish:
                    new_signal = -SIZE_ENTRY
        
        # === MEAN REVERSION MODE ===
        elif is_ranging:
            # Long: RSI oversold + price below HMA21 + weekly not bearish
            if rsi_oversold and close[i] < hma_21[i]:
                if not weekly_bearish:
                    new_signal = SIZE_ENTRY
            
            # Short: RSI overbought + price above HMA21 + weekly not bullish
            elif rsi_overbought and close[i] > hma_21[i]:
                if not weekly_bullish:
                    new_signal = -SIZE_ENTRY
        
        # === BREAKOUT MODE (always available) ===
        if new_signal == 0.0:
            # Strong bullish breakout
            if hma_cross_bullish and close[i] > sma_200[i]:
                new_signal = SIZE_ENTRY
            # Strong bearish breakout
            elif hma_cross_bearish and close[i] < sma_200[i]:
                new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from highest)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from lowest)
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = entry_price - close[i]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reduced (take profit)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals