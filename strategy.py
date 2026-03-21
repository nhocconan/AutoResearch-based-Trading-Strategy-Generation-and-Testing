#!/usr/bin/env python3
"""
Experiment #127: 15m Fisher Transform + 4h HMA Trend + Choppiness Filter
Hypothesis: 15m strategies failed because they traded too much in choppy markets.
This strategy uses Choppiness Index (CHOP) to ONLY trade when market is trending (CHOP<50).
Fisher Transform catches reversals better than RSI in trending markets.
4h HMA provides strong trend filter - only trade in direction of HTF trend.
Volume confirmation via taker_buy_ratio ensures institutional participation.
Fewer, higher-quality trades should reduce fee drag and improve Sharpe.
Position sizing: 0.25 entry, 0.15 at 2R profit, stoploss at 2.5*ATR trailing.
Timeframe: 15m with 4h HTF filter balances signal frequency vs noise.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_fisher_4h_hma_chop_volume_v1"
timeframe = "15m"
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
    half = period // 2
    if half < 1:
        half = 1
    sqrt_period = int(np.sqrt(period))
    if sqrt_period < 1:
        sqrt_period = 1
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into a Gaussian normal distribution for better reversal signals.
    """
    hl2 = (high + low) / 2
    hl2_s = pd.Series(hl2)
    
    # Normalize price within recent range
    highest = hl2_s.rolling(window=period, min_periods=period).max()
    lowest = hl2_s.rolling(window=period, min_periods=period).min()
    
    # Avoid division by zero
    range_hl = highest - lowest
    range_hl = np.where(range_hl == 0, 0.001, range_hl)
    
    # Normalized value between 0 and 1, then scaled to -0.99 to +0.99
    norm = 0.999 * (hl2 - lowest) / range_hl + 0.001
    norm = np.clip(norm, 0.001, 0.999)
    
    # Fisher transformation
    fisher = 0.5 * np.log((1 + norm) / (1 - norm))
    fisher_s = pd.Series(fisher)
    
    # Signal line (previous fisher value)
    fisher_signal = fisher_s.shift(1).values
    
    return fisher_s.values, fisher_signal

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market (avoid trend trades)
    CHOP < 38.2 = trending market (good for trend trades)
    We use threshold of 50 as middle ground for 15m timeframe.
    """
    n = len(close)
    chop = np.zeros(n)
    
    for i in range(period, n):
        # Highest high and lowest low over period
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        
        # Sum of ATR over period
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr1 = high[j] - low[j]
            tr2 = np.abs(high[j] - close[j-1]) if j > 0 else tr1
            tr3 = np.abs(low[j] - close[j-1]) if j > 0 else tr1
            tr_sum += np.maximum(tr1, np.maximum(tr2, tr3))
        
        # CHOP formula
        if hh > ll and tr_sum > 0:
            chop[i] = 100 * np.log10((hh - ll) / tr_sum) / np.log10(period)
        else:
            chop[i] = 50.0  # neutral
    
    return chop

def calculate_taker_buy_ratio(volume, taker_buy_volume):
    """Calculate taker buy volume ratio (institutional buying pressure)."""
    ratio = np.zeros(len(volume))
    mask = volume > 0
    ratio[mask] = taker_buy_volume[mask] / volume[mask]
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_volume = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    chop = calculate_choppiness_index(high, low, close, 14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, 9)
    taker_ratio = calculate_taker_buy_ratio(volume, taker_buy_volume)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # HTF trend filter (4h HMA)
        hma_4h_valid = not np.isnan(hma_4h_aligned[i]) and hma_4h_aligned[i] > 0
        trend_bullish = hma_4h_valid and close[i] > hma_4h_aligned[i]
        trend_bearish = hma_4h_valid and close[i] < hma_4h_aligned[i]
        
        # Choppiness filter - only trade in trending markets
        trending_market = chop[i] < 50.0  # Below 50 = trending
        ranging_market = chop[i] >= 50.0  # Above 50 = choppy (avoid)
        
        # Fisher Transform signals
        fisher_valid = not np.isnan(fisher[i]) and not np.isnan(fisher_signal[i])
        fisher_long = fisher_valid and fisher[i] > fisher_signal[i] and fisher_signal[i] < -0.5
        fisher_short = fisher_valid and fisher[i] < fisher_signal[i] and fisher_signal[i] > 0.5
        
        # Fisher extreme levels (reversal zones)
        fisher_oversold = fisher_valid and fisher[i] < -1.5
        fisher_overbought = fisher_valid and fisher[i] > 1.5
        
        # Volume confirmation
        volume_bullish = taker_ratio[i] > 0.52
        volume_bearish = taker_ratio[i] < 0.48
        
        new_signal = 0.0
        
        # LONG ENTRY conditions
        # Path 1: Trending market + 4h bullish + Fisher crossover from oversold
        if trending_market and trend_bullish and fisher_long and fisher_oversold:
            new_signal = SIZE_ENTRY
        # Path 2: Trending market + 4h bullish + Fisher crossover + Volume
        elif trending_market and trend_bullish and fisher_long and volume_bullish:
            new_signal = SIZE_ENTRY
        # Path 3: Strong Fisher reversal + 4h trend alignment
        elif trending_market and trend_bullish and fisher_oversold and fisher[i] > fisher_signal[i]:
            new_signal = SIZE_ENTRY
        
        # SHORT ENTRY conditions
        # Path 1: Trending market + 4h bearish + Fisher crossover from overbought
        if trending_market and trend_bearish and fisher_short and fisher_overbought:
            new_signal = -SIZE_ENTRY
        # Path 2: Trending market + 4h bearish + Fisher crossover + Volume
        elif trending_market and trend_bearish and fisher_short and volume_bearish:
            new_signal = -SIZE_ENTRY
        # Path 3: Strong Fisher reversal + 4h trend alignment
        elif trending_market and trend_bearish and fisher_overbought and fisher[i] < fisher_signal[i]:
            new_signal = -SIZE_ENTRY
        
        # Stoploss logic (Rule 6) - check BEFORE updating position tracking
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