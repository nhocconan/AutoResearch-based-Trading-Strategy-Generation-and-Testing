#!/usr/bin/env python3
"""
Experiment #307: 15m Multi-HTF Supertrend with Volatility Squeeze Breakout

Hypothesis: 15m has failed in previous experiments (#295, #301) because:
1. Too much noise without strong HTF filtering
2. Mean reversion doesn't work on crypto perpetuals
3. Complex ensembles create conflicting signals

This strategy uses TRIPLE HTF FILTER + volatility squeeze breakout:
1. 1d HMA(21) = meta-trend direction (strongest filter)
2. 4h HMA(21) = intermediate trend confirmation
3. 1h Supertrend(10,3) = momentum alignment
4. 15m Bollinger Band squeeze = low-volatility coiling before breakout
5. 15m price breakout above/below BB + volume confirmation = entry trigger

Why this might work on 15m:
- Triple HTF filter (1d+4h+1h) eliminates 15m noise
- BB squeeze detects consolidation before explosive moves
- Volume confirmation ensures real breakouts, not fakeouts
- Tighter stoploss (2.0*ATR) appropriate for 15m timeframe
- Fewer but higher-quality trades (target 30-50/year)

Timeframe: 15m (REQUIRED for this experiment)
HTF: 1h, 4h, 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 discrete levels (conservative for 15m)
Stoploss: 2.0 * ATR(14) trailing (tighter than 4h/12h strategies)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_supertrend_triple_htf_bb_squeeze_volume_atr_v1"
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
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Calculate Supertrend indicator.
    Returns: supertrend_values, supertrend_direction (1=bullish, -1=bearish)
    """
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    # Basic Upper and Lower Bands
    hl2 = (high + low) / 2.0
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    # Supertrend values
    supertrend = np.zeros(n)
    direction = np.zeros(n)  # 1 = bullish (price above ST), -1 = bearish
    
    supertrend[0] = upper_band[0]
    direction[0] = -1
    
    for i in range(1, n):
        if close[i] > supertrend[i-1]:
            # Bullish: use lower band
            supertrend[i] = lower_band[i]
            direction[i] = 1
        else:
            # Bearish: use upper band
            supertrend[i] = upper_band[i]
            direction[i] = -1
    
    return supertrend, direction

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper.values, lower.values, sma.values

def calculate_bb_width(upper, lower, sma):
    """Calculate Bollinger Band Width (volatility measure)."""
    width = (upper - lower) / sma
    return width

def calculate_bb_percentile(close, upper, lower, lookback=50):
    """
    Calculate where price is within BB range (0=lower, 1=upper).
    Also detect squeeze: BB width at lowest 20% of recent range.
    """
    n = len(close)
    bb_pct = np.zeros(n)
    bb_pct[:] = np.nan
    squeeze = np.zeros(n, dtype=bool)
    
    for i in range(lookback, n):
        if not np.isnan(upper[i]) and not np.isnan(lower[i]) and upper[i] != lower[i]:
            bb_pct[i] = (close[i] - lower[i]) / (upper[i] - lower[i])
        
        # Squeeze detection: current BB width vs recent 50-bar range
        width = (upper[i] - lower[i]) / sma[i] if not np.isnan(sma[i]) and sma[i] != 0 else 0
        recent_widths = []
        for j in range(max(0, i-lookback), i):
            if not np.isnan(upper[j]) and not np.isnan(lower[j]) and not np.isnan(sma[j]) and sma[j] != 0:
                w = (upper[j] - lower[j]) / sma[j]
                if not np.isnan(w):
                    recent_widths.append(w)
        
        if len(recent_widths) >= 20:
            width_percentile = np.percentile(recent_widths, 20)
            squeeze[i] = width < width_percentile
    
    return bb_pct, squeeze

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    _, supertrend_1h_dir = calculate_supertrend(df_1h['high'].values, df_1h['low'].values, df_1h['close'].values, 10, 3.0)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    supertrend_1h_aligned = align_htf_to_ltf(prices, df_1h, supertrend_1h_dir)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    bb_upper, bb_lower, bb_sma = calculate_bollinger_bands(close, 20, 2.0)
    bb_pct, bb_squeeze = calculate_bb_percentile(close, bb_upper, bb_lower, 50)
    
    # Volume moving average for confirmation
    volume_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.20  # Conservative base for 15m
    SIZE_STRONG = 0.30  # Increased size in strong setup
    
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
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(supertrend_1h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME BIAS (Triple HTF Filter) ===
        # 1d HMA = meta-trend (strongest filter)
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # 4h HMA = intermediate trend confirmation
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # 1h Supertrend = momentum alignment
        bull_momentum_1h = supertrend_1h_aligned[i] > 0
        bear_momentum_1h = supertrend_1h_aligned[i] < 0
        
        # === VOLATILITY SQUEEZE DETECTION ===
        # Only trade after consolidation (BB squeeze)
        in_squeeze = bb_squeeze[i]
        
        # === BREAKOUT DETECTION ===
        # Price breaks above BB upper = bullish breakout
        bb_bullish_breakout = close[i] > bb_upper[i-1] if not np.isnan(bb_upper[i-1]) else False
        # Price breaks below BB lower = bearish breakout
        bb_bearish_breakout = close[i] < bb_lower[i-1] if not np.isnan(bb_lower[i-1]) else False
        
        # === VOLUME CONFIRMATION ===
        # Volume must be above average for breakout confirmation
        volume_confirmed = volume[i] > 1.2 * volume_sma[i] if not np.isnan(volume_sma[i]) else False
        
        # === ENTRY CONDITIONS ===
        new_signal = 0.0
        position_size = SIZE_BASE
        
        # LONG ENTRY: Need ALL HTF aligned bullish + BB squeeze + breakout + volume
        # Stricter conditions for 15m to reduce noise
        long_conditions = (
            bull_trend_1d and  # 1d HMA meta-trend bullish
            bull_trend_4h and  # 4h HMA intermediate bullish
            bull_momentum_1h and  # 1h Supertrend bullish
            bb_bullish_breakout and  # BB upper breakout
            volume_confirmed  # Volume confirmation
        )
        
        # SHORT ENTRY: Mirror of long
        short_conditions = (
            bear_trend_1d and  # 1d HMA meta-trend bearish
            bear_trend_4h and  # 4h HMA intermediate bearish
            bear_momentum_1h and  # 1h Supertrend bearish
            bb_bearish_breakout and  # BB lower breakout
            volume_confirmed  # Volume confirmation
        )
        
        # Increase size if squeeze was present (higher probability setup)
        if (long_conditions or short_conditions) and in_squeeze:
            position_size = SIZE_STRONG
        
        # === GENERATE SIGNAL ===
        if long_conditions:
            new_signal = position_size
        
        if short_conditions:
            new_signal = -position_size
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing (tighter for 15m) ===
        # Check stoploss on EXISTING position before considering new entry
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                # Trailing stop: 2.0 * ATR below highest close (tighter for 15m)
                stoploss_price = highest_close - 2.0 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                # Trailing stop: 2.0 * ATR above lowest close
                stoploss_price = lowest_close + 2.0 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
        
        # === HTF TREND REVERSAL EXIT ===
        # Exit if ANY HTF bias reverses against position (strict filter)
        if in_position and new_signal != 0.0:
            if position_side > 0 and (bear_trend_1d or bear_trend_4h or bear_momentum_1h):
                new_signal = 0.0  # HTF trend reversed against long
            if position_side < 0 and (bull_trend_1d or bull_trend_4h or bull_momentum_1h):
                new_signal = 0.0  # HTF trend reversed against short
        
        # === BB REVERSAL EXIT ===
        # Exit if BB breaks against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bb_bearish_breakout:
                new_signal = 0.0  # BB broke against long
            if position_side < 0 and bb_bullish_breakout:
                new_signal = 0.0  # BB broke against short
        
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
            # else: maintaining same position direction (possibly adjusted size)
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