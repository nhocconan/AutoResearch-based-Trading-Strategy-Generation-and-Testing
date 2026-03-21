#!/usr/bin/env python3
"""
Experiment #170: 30m RSI Mean Reversion with 4h HMA Trend Filter
Hypothesis: 30m timeframe captures intraday swings while 4h HMA provides 
major trend bias. RSI(7) extremes (25/75) with volume confirmation generate 
frequent but filtered entries. Simpler than regime-detection approaches that 
failed in exp#164/169. ATR stoploss at 2.0*ATR limits drawdown. Position 
sizing 0.30 entry, 0.15 half-profit to reduce fee churn while maintaining 
exposure. This targets the 2022 crash (short on 4h bearish) and 2025 
consolidation (mean reversion longs/shorts).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_rsi_4h_hma_volume_atr_v1"
timeframe = "30m"
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

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs rolling average."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean().values
    vol_avg = np.where(vol_avg > 0, vol_avg, 1e-10)
    vol_ratio = volume / vol_avg
    vol_ratio = np.where(np.isnan(vol_ratio), 1.0, vol_ratio)
    return vol_ratio

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    std = np.where(std > 0, std, 1e-10)
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

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
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi_fast = calculate_rsi(close, 7)  # Faster RSI for 30m
    rsi_slow = calculate_rsi(close, 14)
    vol_ratio = calculate_volume_ratio(volume, 20)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    hma_20 = calculate_hma(close, 20)
    hma_50 = calculate_hma(close, 50)
    
    # BB bandwidth percentile for regime
    bb_bw = (bb_upper - bb_lower) / bb_mid
    bb_bw = np.where(np.isnan(bb_bw), 0.0, bb_bw)
    bb_percentile = pd.Series(bb_bw).rolling(window=100, min_periods=50).apply(
        lambda x: np.percentile(x, 50), raw=True
    ).values
    bb_percentile = np.where(np.isnan(bb_percentile), 50.0, bb_percentile)
    
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
    
    for i in range(100, n):
        # 4h trend filter
        hma_4h_valid = hma_4h_aligned[i] > 0
        trend_bullish_4h = hma_4h_valid and close[i] > hma_4h_aligned[i]
        trend_bearish_4h = hma_4h_valid and close[i] < hma_4h_aligned[i]
        
        # 30m trend
        trend_bullish_30m = hma_20[i] > hma_50[i]
        trend_bearish_30m = hma_20[i] < hma_50[i]
        
        # RSI signals (loosened for more trades)
        rsi_oversold = rsi_fast[i] < 30
        rsi_overbought = rsi_fast[i] > 70
        rsi_extreme_oversold = rsi_fast[i] < 20
        rsi_extreme_overbought = rsi_fast[i] > 80
        rsi_rising = rsi_fast[i] > rsi_fast[i-2] if i > 2 else False
        rsi_falling = rsi_fast[i] < rsi_fast[i-2] if i > 2 else False
        
        # Volume confirmation
        volume_high = vol_ratio[i] > 1.2
        
        # BB position
        near_lower_bb = close[i] < bb_lower[i] * 1.005
        near_upper_bb = close[i] > bb_upper[i] * 0.995
        bb_expanded = bb_bw[i] > bb_percentile[i]
        bb_squeezed = bb_bw[i] < bb_percentile[i]
        
        new_signal = 0.0
        
        # === LONG ENTRIES ===
        # Mean reversion long: RSI oversold + near lower BB + volume
        if rsi_oversold and near_lower_bb:
            if volume_high or rsi_rising:
                # Only if 4h not strongly bearish
                if not trend_bearish_4h or rsi_extreme_oversold:
                    new_signal = SIZE_ENTRY
        
        # Trend pullback long: 4h bullish + 30m pullback
        elif trend_bullish_4h and trend_bullish_30m:
            if rsi_fast[i] < 45 and rsi_rising:
                if volume_high or near_lower_bb:
                    new_signal = SIZE_ENTRY
        
        # Breakout long: BB squeeze + volume + 4h bullish
        elif bb_squeezed and close[i] > bb_mid[i]:
            if volume_high and (trend_bullish_4h or trend_bullish_30m):
                new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES ===
        # Mean reversion short: RSI overbought + near upper BB + volume
        if new_signal == 0.0 and rsi_overbought and near_upper_bb:
            if volume_high or rsi_falling:
                # Only if 4h not strongly bullish
                if not trend_bullish_4h or rsi_extreme_overbought:
                    new_signal = -SIZE_ENTRY
        
        # Trend pullback short: 4h bearish + 30m pullback
        elif new_signal == 0.0 and trend_bearish_4h and trend_bearish_30m:
            if rsi_fast[i] > 55 and rsi_falling:
                if volume_high or near_upper_bb:
                    new_signal = -SIZE_ENTRY
        
        # Breakout short: BB squeeze + volume + 4h bearish
        elif new_signal == 0.0 and bb_squeezed and close[i] < bb_mid[i]:
            if volume_high and (trend_bearish_4h or trend_bearish_30m):
                new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.0*ATR from highest)
            current_stop = highest_close - 2.0 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.0 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.0*ATR from lowest)
            current_stop = lowest_close + 2.0 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.0 * atr[i]
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
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
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