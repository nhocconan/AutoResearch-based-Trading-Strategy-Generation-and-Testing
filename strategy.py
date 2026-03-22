#!/usr/bin/env python3
"""
Experiment #017: 12h Asymmetric Mean Reversion with 1d HTF Trend Filter
Hypothesis: 12h timeframe captures multi-day swings while avoiding noise.
Using asymmetric logic: Long on oversold conditions in bull/neutral regimes,
Short on overbought conditions in bear/neutral regimes. 1d HMA for primary
trend bias. Donchian breakout as secondary entry path. Conservative sizing
(0.25-0.30) with 2.5*ATR stop (wider for 12h). Multiple entry paths ensure
>=10 trades per symbol. Avoids complex regime filters that failed previously.
Timeframe: 12h (REQUIRED), HTF: 1d
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_asym_meanrev_1d_hma_donchian_rsi_zscore_atr_v1"
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

def calculate_zscore(close, period=20):
    """Calculate Z-score of price relative to rolling mean."""
    close_s = pd.Series(close)
    mean = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    zscore = (close - mean) / (std + 1e-10)
    return zscore

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high, lowest low over period)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    mid = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    return upper, mid, lower

def calculate_volume_ratio(taker_buy_volume, volume):
    """Calculate taker buy volume ratio (0-1, >0.5 = bullish pressure)."""
    ratio = np.zeros(len(volume))
    mask = volume > 0
    ratio[mask] = taker_buy_volume[mask] / volume[mask]
    return ratio

def calculate_kama(close, period=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average."""
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Efficiency Ratio
    er = np.zeros(n)
    for i in range(period, n):
        change = np.abs(close[i] - close[i - period])
        noise = np.sum(np.abs(np.diff(close[i-period:i+1])))
        if noise > 0:
            er[i] = change / noise
    
    # Smoothing constant
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    
    # First valid KAMA
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    return kama

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_vol = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    rsi_fast = calculate_rsi(close, 7)
    zscore = calculate_zscore(close, 20)
    vol_ratio = calculate_volume_ratio(taker_buy_vol, volume)
    
    # Donchian Channel
    donch_upper, donch_lower = calculate_donchian(high, low, 20)
    
    # Bollinger Bands
    bb_upper, bb_mid, bb_lower = calculate_bollinger(close, 20, 2.0)
    
    # HMA for 12h trend
    hma_12h = calculate_hma(close, 21)
    hma_12h_fast = calculate_hma(close, 10)
    
    # KAMA for adaptive trend
    kama_12h = calculate_kama(close, 10)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.28
    SIZE_HALF = 0.14
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(zscore[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(donch_upper[i]):
            signals[i] = 0.0
            continue
        
        # 1d trend bias (HTF) - use slope for stronger signal
        htf_bullish = close[i] > hma_1d_aligned[i]
        htf_bearish = close[i] < hma_1d_aligned[i]
        
        # Check 1d HMA slope (trend direction)
        htf_slope_up = False
        htf_slope_down = False
        if i >= 48 and not np.isnan(hma_1d_aligned[i-48]):  # 48 x 12h = 1 day lookback
            htf_slope_up = hma_1d_aligned[i] > hma_1d_aligned[i-48]
            htf_slope_down = hma_1d_aligned[i] < hma_1d_aligned[i-48]
        
        # 12h trend
        hma_12h_bullish = close[i] > hma_12h[i]
        hma_12h_bearish = close[i] < hma_12h[i]
        
        # HMA crossover
        fast_above_slow = hma_12h_fast[i] > hma_12h[i]
        fast_below_slow = hma_12h_fast[i] < hma_12h[i]
        
        # KAMA trend
        kama_bullish = close[i] > kama_12h[i] if not np.isnan(kama_12h[i]) else False
        kama_bearish = close[i] < kama_12h[i] if not np.isnan(kama_12h[i]) else False
        
        # Donchian breakout (use previous bar to avoid look-ahead)
        breakout_long = close[i] > donch_upper[i-1] if i > 0 else False
        breakout_short = close[i] < donch_lower[i-1] if i > 0 else False
        
        # Volume confirmation
        vol_bullish = vol_ratio[i] > 0.50
        vol_bearish = vol_ratio[i] < 0.50
        
        # Z-score extremes
        zscore_oversold = zscore[i] < -1.5
        zscore_overbought = zscore[i] > 1.5
        zscore_extreme_oversold = zscore[i] < -2.0
        zscore_extreme_overbought = zscore[i] > 2.0
        
        # RSI extremes
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_fast_oversold = rsi_fast[i] < 30
        rsi_fast_overbought = rsi_fast[i] > 70
        
        # Bollinger Band position
        price_near_lower = close[i] < bb_lower[i] * 1.01
        price_near_upper = close[i] > bb_upper[i] * 0.99
        price_breaks_lower = close[i] < bb_lower[i]
        price_breaks_upper = close[i] > bb_upper[i]
        
        new_signal = 0.0
        
        # === ASYMMETRIC LONG ENTRIES ===
        # Long entries favor oversold conditions + bullish/bearish HTF filter
        
        # Path 1: Z-score extreme oversold + 1d bullish (deep pullback in uptrend)
        if zscore_extreme_oversold and htf_bullish:
            new_signal = SIZE_ENTRY
        
        # Path 2: RSI oversold + 1d bullish + 12h HMA bullish (dip buy in uptrend)
        elif rsi_oversold and htf_bullish and hma_12h_bullish:
            new_signal = SIZE_ENTRY
        
        # Path 3: Z-score oversold + RSI oversold (double confirmation mean reversion)
        elif zscore_oversold and rsi_oversold:
            new_signal = SIZE_ENTRY
        
        # Path 4: Price at BB lower + 1d bullish + RSI fast oversold
        elif price_near_lower and htf_bullish and rsi_fast_oversold:
            new_signal = SIZE_ENTRY
        
        # Path 5: Donchian breakout + 1d bullish + volume (trend continuation)
        elif breakout_long and htf_bullish and vol_bullish:
            new_signal = SIZE_ENTRY
        
        # Path 6: HMA crossover up + 1d bullish (momentum entry)
        elif fast_above_slow and htf_bullish and hma_12h_bullish:
            new_signal = SIZE_ENTRY
        
        # Path 7: KAMA bullish + 1d slope up + RSI not overbought
        elif kama_bullish and htf_slope_up and rsi[i] < 60:
            new_signal = SIZE_ENTRY
        
        # === ASYMMETRIC SHORT ENTRIES ===
        # Short entries favor overbought conditions + bearish/bullish HTF filter
        
        # Path 1: Z-score extreme overbought + 1d bearish (rally in downtrend)
        if zscore_extreme_overbought and htf_bearish:
            new_signal = -SIZE_ENTRY
        
        # Path 2: RSI overbought + 1d bearish + 12h HMA bearish (rally sell in downtrend)
        elif rsi_overbought and htf_bearish and hma_12h_bearish:
            new_signal = -SIZE_ENTRY
        
        # Path 3: Z-score overbought + RSI overbought (double confirmation mean reversion)
        elif zscore_overbought and rsi_overbought:
            new_signal = -SIZE_ENTRY
        
        # Path 4: Price at BB upper + 1d bearish + RSI fast overbought
        elif price_near_upper and htf_bearish and rsi_fast_overbought:
            new_signal = -SIZE_ENTRY
        
        # Path 5: Donchian breakdown + 1d bearish + volume (trend continuation)
        elif breakout_short and htf_bearish and vol_bearish:
            new_signal = -SIZE_ENTRY
        
        # Path 6: HMA crossover down + 1d bearish (momentum entry)
        elif fast_below_slow and htf_bearish and hma_12h_bearish:
            new_signal = -SIZE_ENTRY
        
        # Path 7: KAMA bearish + 1d slope down + RSI not oversold
        elif kama_bearish and htf_slope_down and rsi[i] > 40:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for 12h timeframe - wider stops)
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
            
            # Calculate trailing stop (2.5*ATR for 12h timeframe)
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
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
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