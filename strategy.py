#!/usr/bin/env python3
"""
Experiment #189: 1h Fisher Transform + 4h HMA Trend with Volume Confirmation
Hypothesis: Fisher Transform excels at catching reversals in bear/range markets (2022, 2025)
while 4h HMA provides trend bias to avoid counter-trend trades. Volume confirmation
filters false breakouts. Adaptive position sizing reduces exposure during high volatility.
This addresses the failure of pure trend-following (destroyed in 2022 crash) and pure
mean-reversion (fails in strong trends). Fisher crosses are more frequent than RSI extremes,
ensuring sufficient trades (>10 on train, >3 on test).
Position sizing: 0.25 base, reduced to 0.15 in high vol regimes. Stoploss at 2.5*ATR.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_4h_hma_volume_adaptive_v1"
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
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into a Gaussian normal distribution for clearer reversal signals.
    Long when Fisher crosses above -1.5, short when crosses below +1.5.
    Reference: John F. Ehlers, "Cybernetic Analysis for Stocks and Futures"
    """
    hl2 = (high + low) / 2.0
    hl2_s = pd.Series(hl2)
    
    # Calculate EMA of HL2
    ema_hl2 = hl2_s.ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Normalize to -1 to +1 range
    highest = pd.Series(hl2).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(hl2).rolling(window=period, min_periods=period).min().values
    range_hl = highest - lowest
    range_hl = np.where(range_hl > 0, range_hl, 1e-10)
    
    normalized = 2.0 * (hl2 - lowest) / range_hl - 1.0
    normalized = np.clip(normalized, -0.999, 0.999)
    
    # Fisher transform
    fisher = 0.5 * np.log((1 + normalized) / (1 - normalized))
    fisher = np.where(np.isnan(fisher), 0.0, fisher)
    
    # Signal line (EMA of Fisher)
    fisher_s = pd.Series(fisher)
    fisher_signal = fisher_s.ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return fisher, fisher_signal

def calculate_volume_ma(volume, period=20):
    """Calculate volume moving average for volume confirmation."""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_ma

def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Directional Movement
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / np.where(atr > 0, atr, 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / np.where(atr > 0, atr, 1e-10)
    
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) > 0, plus_di + minus_di, 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def calculate_volatility_percentile(atr, lookback=100):
    """Calculate ATR percentile for adaptive position sizing."""
    atr_s = pd.Series(atr)
    atr_percentile = atr_s.rolling(window=lookback, min_periods=50).apply(
        lambda x: np.percentile(x, 50), raw=True
    ).values
    atr_percentile = np.where(np.isnan(atr_percentile), 50.0, atr_percentile)
    return atr_percentile

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, 9)
    vol_ma = calculate_volume_ma(volume, 20)
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    hma_20 = calculate_hma(close, 20)
    hma_50 = calculate_hma(close, 50)
    atr_pct = calculate_volatility_percentile(atr, 100)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_LOW_VOL = 0.30
    SIZE_HIGH_VOL = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # HTF trend filter (4h HMA)
        hma_4h_valid = hma_4h_aligned[i] > 0
        trend_4h_bullish = hma_4h_valid and close[i] > hma_4h_aligned[i]
        trend_4h_bearish = hma_4h_valid and close[i] < hma_4h_aligned[i]
        
        # 1h trend
        trend_1h_bullish = hma_20[i] > hma_50[i]
        trend_1h_bearish = hma_20[i] < hma_50[i]
        
        # Fisher Transform signals
        fisher_cross_up = fisher[i] > fisher_signal[i] and fisher[i-1] <= fisher_signal[i-1]
        fisher_cross_down = fisher[i] < fisher_signal[i] and fisher[i-1] >= fisher_signal[i-1]
        fisher_oversold = fisher[i] < -1.5
        fisher_overbought = fisher[i] > 1.5
        
        # Volume confirmation
        volume_above_avg = volume[i] > vol_ma[i] * 1.2 if vol_ma[i] > 0 else False
        
        # ADX trend strength
        trend_strong = adx[i] > 20
        trend_weak = adx[i] < 25
        
        # Volatility regime for adaptive sizing
        high_volatility = atr_pct[i] > 60
        low_volatility = atr_pct[i] < 40
        
        # Determine position size based on volatility
        if high_volatility:
            size = SIZE_HIGH_VOL
        elif low_volatility:
            size = SIZE_LOW_VOL
        else:
            size = SIZE_BASE
        
        new_signal = 0.0
        
        # === LONG ENTRIES ===
        # Condition 1: Fisher oversold + 4h trend bullish (pullback in uptrend)
        if fisher_oversold and trend_4h_bullish:
            if fisher_cross_up or volume_above_avg:
                new_signal = size
        
        # Condition 2: Fisher cross up + 1h trend bullish + volume confirmation
        elif fisher_cross_up and trend_1h_bullish:
            if volume_above_avg or trend_strong:
                new_signal = size
        
        # Condition 3: HMA crossover + Fisher confirmation
        elif hma_20[i] > hma_50[i] and hma_20[i-1] <= hma_50[i-1]:
            if fisher[i] > fisher_signal[i] or volume_above_avg:
                new_signal = size
        
        # === SHORT ENTRIES ===
        # Condition 1: Fisher overbought + 4h trend bearish (rally in downtrend)
        if fisher_overbought and trend_4h_bearish:
            if fisher_cross_down or volume_above_avg:
                new_signal = -size
        
        # Condition 2: Fisher cross down + 1h trend bearish + volume confirmation
        elif fisher_cross_down and trend_1h_bearish:
            if volume_above_avg or trend_strong:
                new_signal = -size
        
        # Condition 3: HMA crossover + Fisher confirmation
        elif hma_20[i] < hma_50[i] and hma_20[i-1] >= hma_50[i-1]:
            if fisher[i] < fisher_signal[i] or volume_above_avg:
                new_signal = -size
        
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
                    new_signal = size / 2
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
                    new_signal = -size / 2
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