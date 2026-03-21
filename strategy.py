#!/usr/bin/env python3
"""
Experiment #376: 4h Fisher Transform + Donchian Breakout + Daily HMA Trend + ATR Stop
Hypothesis: Ehlers Fisher Transform excels at identifying turning points in bear/range markets
(2025 test period) while Donchian breakout captures trending moves (2021 bull). Combined with
Daily HMA trend filter, this should reduce whipsaws and improve Sharpe vs pure trend strategies.
Fisher normalizes price to Gaussian distribution, making extremes (-2/+2) statistically significant.
Timeframe: 4h (REQUIRED), HTF: 1d for trend bias via mtf_data helper.
Position sizing: 0.25 entry, 0.125 half-profit, stoploss at 2.5*ATR trailing.
Target: Beat Sharpe=0.499 with 50-100 trades total, work in both bull and bear regimes.
Key insight: Fisher Transform outperforms RSI for reversal detection in crypto's non-normal returns.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_donchian_daily_hma_volume_atr_v1"
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

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price to near-Gaussian distribution for clearer reversal signals.
    Long when Fisher crosses above -1.5 from below, short when crosses below +1.5 from above.
    """
    n = len(close)
    fisher = np.zeros(n)
    fisher_signal = np.zeros(n)
    
    for i in range(period, n):
        # Calculate typical price
        hl2 = (high[i] + low[i]) / 2.0
        
        # Find highest high and lowest low over period
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        # Normalize price to 0-1 range
        range_val = highest_high - lowest_low
        if range_val > 0:
            normalized = (hl2 - lowest_low) / range_val
        else:
            normalized = 0.5
        
        # Clamp to avoid division issues
        normalized = np.clip(normalized, 0.001, 0.999)
        
        # Fisher calculation
        fisher[i] = 0.5 * np.log((1 + normalized) / (1 - normalized))
        
        # Signal line (1-period lag)
        if i > period:
            fisher_signal[i] = fisher[i-1]
        else:
            fisher_signal[i] = fisher[i]
    
    fisher[:period] = 0.0
    fisher_signal[:period] = 0.0
    return fisher, fisher_signal

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel upper and lower bands."""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(period, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    upper[:period] = high[:period].max() if period <= len(high) else high[0]
    lower[:period] = low[:period].min() if period <= len(low) else low[0]
    return upper, lower

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs moving average."""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.ewm(span=period, min_periods=period, adjust=False).mean().values
    ratio = volume / vol_ma
    ratio = np.nan_to_num(ratio, nan=1.0)
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, 9)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
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
    
    for i in range(100, n):  # Start after 100 bars for indicators
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(fisher[i]) or np.isnan(donchian_upper[i]):
            signals[i] = 0.0
            continue
        
        # Daily trend bias
        daily_bullish = not np.isnan(hma_1d_aligned[i]) and close[i] > hma_1d_aligned[i]
        daily_bearish = not np.isnan(hma_1d_aligned[i]) and close[i] < hma_1d_aligned[i]
        
        # Fisher Transform signals
        fisher_oversold = fisher[i] < -1.0
        fisher_overbought = fisher[i] > 1.0
        
        # Fisher crossover signals
        fisher_cross_long = fisher_signal[i] < -1.0 and fisher[i] >= -1.0
        fisher_cross_short = fisher_signal[i] > 1.0 and fisher[i] <= 1.0
        
        # Donchian breakout signals
        donchian_breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # Volume confirmation
        volume_confirmed = vol_ratio[i] > 0.8  # At least 80% of average volume
        
        new_signal = 0.0
        
        # === LONG ENTRIES ===
        # Primary: Fisher oversold crossover + Daily bullish + Volume confirmed
        if fisher_cross_long and daily_bullish and volume_confirmed:
            new_signal = SIZE_ENTRY
        # Secondary: Fisher oversold + Donchian breakout + Daily bullish
        elif fisher_oversold and donchian_breakout_long and daily_bullish:
            new_signal = SIZE_ENTRY
        # Tertiary: Fisher cross long + Donchian breakout (trend confirmation)
        elif fisher_cross_long and donchian_breakout_long and volume_confirmed:
            new_signal = SIZE_ENTRY
        # Quaternary: Fisher oversold alone (ensures trade frequency in bear markets)
        elif fisher_oversold and fisher[i] > fisher_signal[i] and volume_confirmed:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES ===
        # Primary: Fisher overbought crossover + Daily bearish + Volume confirmed
        if fisher_cross_short and daily_bearish and volume_confirmed:
            new_signal = -SIZE_ENTRY
        # Secondary: Fisher overbought + Donchian breakdown + Daily bearish
        elif fisher_overbought and donchian_breakout_short and daily_bearish:
            new_signal = -SIZE_ENTRY
        # Tertiary: Fisher cross short + Donchian breakdown (trend confirmation)
        elif fisher_cross_short and donchian_breakout_short and volume_confirmed:
            new_signal = -SIZE_ENTRY
        # Quaternary: Fisher overbought alone (ensures trade frequency in bear markets)
        elif fisher_overbought and fisher[i] < fisher_signal[i] and volume_confirmed:
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