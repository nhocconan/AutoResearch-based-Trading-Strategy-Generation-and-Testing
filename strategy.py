#!/usr/bin/env python3
"""
Experiment #377: 12h Supertrend + Weekly HMA + Bollinger Volatility + RSI Divergence + Volume
Hypothesis: Supertrend provides clean trend signals (proven in #371 baseline). Weekly HMA gives
stronger long-term bias than daily. Bollinger Band Width detects volatility expansion/contraction
(regime filter different from CHOP). RSI divergence catches reversals before price confirms.
Volume spike confirms breakout validity. This combines trend-following (Supertrend) with
momentum confirmation (RSI div) and regime filter (BB Width) for better risk-adjusted returns.
Timeframe: 12h (REQUIRED), HTF: 1w for strong trend bias via mtf_data helper.
Target: Beat Sharpe=0.499 from mtf_12h_supertrend_daily_hma_rsi_pullback_v2
Key insight: Weekly trend filter + volatility regime + divergence detection should reduce
false signals while maintaining trade frequency through multiple entry conditions.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_supertrend_weekly_hma_bbvol_rsi_div_vol_v1"
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

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Calculate Supertrend indicator.
    Returns: supertrend values, direction (1=long, -1=short)
    """
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    hl2 = (high + low) / 2.0
    
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend = np.zeros(n)
    direction = np.zeros(n)
    
    supertrend[0] = upper_band[0]
    direction[0] = 1
    
    for i in range(1, n):
        if close[i-1] <= supertrend[i-1]:
            # Previously short or neutral
            supertrend[i] = min(upper_band[i], supertrend[i-1])
            if close[i] > supertrend[i]:
                direction[i] = 1
                supertrend[i] = lower_band[i]
            else:
                direction[i] = -1
        else:
            # Previously long
            supertrend[i] = max(lower_band[i], supertrend[i-1])
            if close[i] < supertrend[i]:
                direction[i] = -1
                supertrend[i] = upper_band[i]
            else:
                direction[i] = 1
    
    return supertrend, direction

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

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands and Band Width."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    band_width = (upper - lower) / sma * 100.0
    band_width = np.nan_to_num(band_width, nan=0.0)
    return upper, lower, band_width, sma

def calculate_rsi_divergence(close, rsi, lookback=5):
    """
    Detect RSI divergence.
    Bullish: price makes lower low, RSI makes higher low
    Bearish: price makes higher high, RSI makes lower high
    Returns: divergence score (-2 to +2)
    """
    n = len(close)
    divergence = np.zeros(n)
    
    for i in range(lookback * 2, n):
        # Check for bullish divergence (last lookback*2 bars)
        price_low_recent = np.min(close[i-lookback:i])
        price_low_prev = np.min(close[i-lookback*2:i-lookback])
        rsi_low_recent = np.min(rsi[i-lookback:i])
        rsi_low_prev = np.min(rsi[i-lookback*2:i-lookback])
        
        # Bullish: price lower low, RSI higher low
        if price_low_recent < price_low_prev and rsi_low_recent > rsi_low_prev:
            divergence[i] = max(divergence[i], 1.0)
        
        # Check for bearish divergence
        price_high_recent = np.max(close[i-lookback:i])
        price_high_prev = np.max(close[i-lookback*2:i-lookback])
        rsi_high_recent = np.max(rsi[i-lookback:i])
        rsi_high_prev = np.max(rsi[i-lookback*2:i-lookback])
        
        # Bearish: price higher high, RSI lower high
        if price_high_recent > price_high_prev and rsi_high_recent < rsi_high_prev:
            divergence[i] = min(divergence[i], -1.0)
    
    return divergence

def calculate_volume_spike(volume, period=20):
    """Detect volume spikes (> 2x average volume)."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean().values
    vol_spike = volume > 2.0 * vol_avg
    vol_spike = np.nan_to_num(vol_spike, nan=False)
    return vol_spike.astype(float)

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3.0)
    bb_upper, bb_lower, bb_width, bb_sma = calculate_bollinger_bands(close, 20, 2.0)
    rsi_div = calculate_rsi_divergence(close, rsi, 5)
    vol_spike = calculate_volume_spike(volume, 20)
    
    # BB Width percentile for regime (rolling 100 bars)
    bb_width_pct = pd.Series(bb_width).rolling(window=100, min_periods=50).apply(
        lambda x: np.sum(x < x[-1]) / len(x), raw=False
    ).values
    bb_width_pct = np.nan_to_num(bb_width_pct, nan=0.5)
    
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
        if np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(bb_width[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(supertrend[i]) or np.isnan(st_direction[i]):
            signals[i] = 0.0
            continue
        
        # Weekly trend bias
        weekly_bullish = not np.isnan(hma_1w_aligned[i]) and close[i] > hma_1w_aligned[i]
        weekly_bearish = not np.isnan(hma_1w_aligned[i]) and close[i] < hma_1w_aligned[i]
        
        # Volatility regime from BB Width
        is_low_vol = bb_width_pct[i] < 0.3  # Bottom 30% = contraction
        is_high_vol = bb_width_pct[i] > 0.7  # Top 30% = expansion
        
        # Supertrend signals
        st_long = st_direction[i] == 1
        st_short = st_direction[i] == -1
        
        # Supertrend flip detection
        st_flip_long = st_direction[i] == 1 and st_direction[i-1] == -1
        st_flip_short = st_direction[i] == -1 and st_direction[i-1] == 1
        
        # RSI conditions
        rsi_oversold = rsi[i] < 40
        rsi_overbought = rsi[i] > 60
        rsi_neutral = 35 < rsi[i] < 65
        
        # RSI divergence
        rsi_bull_div = rsi_div[i] > 0.5
        rsi_bear_div = rsi_div[i] < -0.5
        
        # Volume confirmation
        vol_confirmed = vol_spike[i] > 0.5
        
        new_signal = 0.0
        
        # === LONG ENTRIES ===
        # Primary: Supertrend flip long + Weekly bullish + Volume spike
        if st_flip_long and weekly_bullish and vol_confirmed:
            new_signal = SIZE_ENTRY
        # Secondary: Supertrend long + Weekly bullish + RSI oversold (pullback entry)
        elif st_long and weekly_bullish and rsi_oversold:
            new_signal = SIZE_ENTRY
        # Tertiary: Supertrend flip long + RSI bull divergence (reversal)
        elif st_flip_long and rsi_bull_div:
            new_signal = SIZE_ENTRY
        # Quaternary: Supertrend long + Low vol regime (trend continuation)
        elif st_long and weekly_bullish and is_low_vol:
            new_signal = SIZE_ENTRY
        # Quintenary: Supertrend long + RSI neutral (momentum continuation)
        elif st_long and rsi_neutral and vol_confirmed:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES ===
        # Primary: Supertrend flip short + Weekly bearish + Volume spike
        if st_flip_short and weekly_bearish and vol_confirmed:
            new_signal = -SIZE_ENTRY
        # Secondary: Supertrend short + Weekly bearish + RSI overbought (pullback entry)
        elif st_short and weekly_bearish and rsi_overbought:
            new_signal = -SIZE_ENTRY
        # Tertiary: Supertrend flip short + RSI bear divergence (reversal)
        elif st_flip_short and rsi_bear_div:
            new_signal = -SIZE_ENTRY
        # Quaternary: Supertrend short + Low vol regime (trend continuation)
        elif st_short and weekly_bearish and is_low_vol:
            new_signal = -SIZE_ENTRY
        # Quintenary: Supertrend short + RSI neutral (momentum continuation)
        elif st_short and rsi_neutral and vol_confirmed:
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