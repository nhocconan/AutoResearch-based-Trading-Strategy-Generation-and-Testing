#!/usr/bin/env python3
"""
Experiment #221: 4h Primary + 1d/1w HTF — Vol Spike Reversion + HMA Trend + Funding Contrarian

Hypothesis: After 12h failures and mixed 4h results, combine three proven edges:
1. VOL SPIKE REVERSION: ATR(7)/ATR(30) > 2.0 indicates panic/extreme vol → mean revert
2. HMA TREND FILTER: 1d HMA(21) for macro bias (aligns with current best strategy)
3. RSI EXTREMES: RSI(7) < 25 or > 75 for entry timing (faster than RSI(14))

Key differences from failed attempts:
1. NO Choppiness Index — vol ratio is simpler and more responsive
2. NO CRSI complexity — use fast RSI(7) for entry timing
3. 1w HMA(14) as secondary macro filter (very slow trend)
4. Asymmetric sizing: full size with HTF trend, half against
5. Looser vol spike threshold (1.8x instead of 2.0x) to ensure trade frequency

TARGET: 30-50 trades/year on 4h, Sharpe > 0.5 on ALL symbols
Position sizing: 0.0, ±0.15, ±0.30 (discrete to minimize fee churn)
Stoploss: ATR(14) 2.5x trailing stop + time-based exit (10 bars)

Research backing:
- Vol spike reversion reported Sharpe 0.8-1.5 through 2022 crash
- Works on BTC/ETH specifically (catches panic bottoms)
- HMA trend filter reduces false signals in strong trends
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_vol_spike_hma_rsi_1d1w_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)) with sqrt(n) window
    Faster and smoother than EMA, less lag.
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    close_s = pd.Series(close)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, period)
    
    hull = 2 * wma_half - wma_full
    hma = wma(hull, sqrt_n)
    
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    n = len(close)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0).values

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper.values, lower.values, sma.values

def calculate_zscore(series, period=20):
    """Calculate Z-score of a series."""
    s = pd.Series(series)
    mean = s.rolling(window=period, min_periods=period).mean()
    std = s.rolling(window=period, min_periods=period).std()
    zscore = (s - mean) / (std + 1e-10)
    return zscore.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 4h indicators (primary timeframe)
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)
    rsi_14 = calculate_rsi(close, period=14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, period=20, std_mult=2.5)
    
    # Vol spike ratio: ATR(7) / ATR(30)
    with np.errstate(divide='ignore', invalid='ignore'):
        vol_spike_ratio = atr_7 / (atr_30 + 1e-10)
    
    # Volume Z-score for confirmation
    vol_zscore = calculate_zscore(volume, period=20)
    
    # Calculate 1d HMA for macro trend (aligned properly)
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 1w HMA for very slow trend (aligned properly)
    hma_1w_raw = calculate_hma(df_1w['close'].values, 14)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE_FULL = 0.30
    POSITION_SIZE_HALF = 0.15
    
    # Track position state for stoploss and time exit
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    bars_in_trade = 0
    
    for i in range(100, n):  # Start later to ensure all indicators ready
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(atr_7[i]) or np.isnan(atr_30[i]):
            continue
        if np.isnan(rsi_7[i]) or np.isnan(rsi_14[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        if np.isnan(vol_spike_ratio[i]):
            continue
        
        # === HTF MACRO BIAS (1d and 1w HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # Strong bullish: both 1d and 1w HMA below price
        macro_bullish = price_above_hma_1d and price_above_hma_1w
        # Strong bearish: both 1d and 1w HMA above price
        macro_bearish = price_below_hma_1d and price_below_hma_1w
        # Mixed/neutral otherwise
        
        # === VOL SPIKE DETECTION ===
        vol_spike = vol_spike_ratio[i] > 1.8  # Lowered from 2.0 for more trades
        vol_extreme = vol_spike_ratio[i] > 2.5  # Very extreme
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi_7[i] < 25.0
        rsi_overbought = rsi_7[i] > 75.0
        rsi_neutral = 35.0 <= rsi_7[i] <= 65.0
        
        # === BOLLINGER POSITION ===
        price_below_bb_lower = close[i] < bb_lower[i]
        price_above_bb_upper = close[i] > bb_upper[i]
        price_near_bb_lower = close[i] < bb_mid[i] - 1.0 * (bb_upper[i] - bb_mid[i])
        price_near_bb_upper = close[i] > bb_mid[i] + 1.0 * (bb_upper[i] - bb_mid[i])
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = vol_zscore[i] > 1.0  # Above average volume
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG ENTRY: Vol spike + RSI oversold + BB lower + HTF bias
        if vol_spike and rsi_oversold:
            if price_below_bb_lower or price_near_bb_lower:
                if macro_bullish:
                    new_signal = POSITION_SIZE_FULL  # With macro trend
                elif price_above_hma_1d:
                    new_signal = POSITION_SIZE_HALF  # Partial macro alignment
                else:
                    new_signal = POSITION_SIZE_HALF  # Counter-trend but extreme
        
        # Also enter on RSI oversold alone if macro strongly bullish
        if rsi_7[i] < 35.0 and macro_bullish and price_near_bb_lower:
            if new_signal == 0.0:
                new_signal = POSITION_SIZE_HALF
        
        # SHORT ENTRY: Vol spike + RSI overbought + BB upper + HTF bias
        if vol_spike and rsi_overbought:
            if price_above_bb_upper or price_near_bb_upper:
                if macro_bearish:
                    new_signal = -POSITION_SIZE_FULL  # With macro trend
                elif price_below_hma_1d:
                    new_signal = -POSITION_SIZE_HALF  # Partial macro alignment
                else:
                    new_signal = -POSITION_SIZE_HALF  # Counter-trend but extreme
        
        # Also enter on RSI overbought alone if macro strongly bearish
        if rsi_7[i] > 65.0 and macro_bearish and price_near_bb_upper:
            if new_signal == 0.0:
                new_signal = -POSITION_SIZE_HALF
        
        # === HOLD POSITION LOGIC ===
        # Hold if in position and conditions still valid
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if RSI not extremely overbought and 1d HMA still supportive
                if rsi_7[i] < 80.0 and price_above_hma_1d:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if RSI not extremely oversold and 1d HMA still supportive
                if rsi_7[i] > 20.0 and price_below_hma_1d:
                    new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === TIME-BASED EXIT (10 bars = 40 hours on 4h) ===
        if in_position:
            bars_in_trade = i - entry_bar
            if bars_in_trade > 10:
                # Exit if RSI moved to neutral (profit taking)
                if position_side > 0 and rsi_7[i] > 55.0:
                    new_signal = 0.0
                elif position_side < 0 and rsi_7[i] < 45.0:
                    new_signal = 0.0
        
        # === MACRO TREND REVERSAL EXIT ===
        # Exit long if 1d HMA turns bearish
        if in_position and position_side > 0 and price_below_hma_1d and rsi_7[i] > 50.0:
            new_signal = 0.0
        
        # Exit short if 1d HMA turns bullish
        if in_position and position_side < 0 and price_above_hma_1d and rsi_7[i] < 50.0:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_bar = i
                bars_in_trade = 0
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_bar = i
                bars_in_trade = 0
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_bar = 0
                bars_in_trade = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals