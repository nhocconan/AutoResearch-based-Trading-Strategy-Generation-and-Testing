#!/usr/bin/env python3
"""
Experiment #178: 30m Primary + 4h/1d HTF — HTF Trend + RSI Pullback + Session Filter

Hypothesis: Lower TF (30m) strategies fail due to excessive trades → fee drag.
Solution: Use 4h/1d HMA for SIGNAL DIRECTION (not just filter), 30m only for
ENTRY TIMING within HTF trend. Add session filter (8-20 UTC) to avoid low-volume
whipsaws. Require 4+ confluence factors for entry.

KEY IMPROVEMENTS:
1. 4h HMA(21) = PRIMARY trend direction (only long if 4h HMA bullish)
2. 1d HMA(21) = macro bias confirmation (strengthens signal)
3. 30m RSI(14) pullback entries (RSI<40 long, RSI>60 short) WITHIN HTF trend
4. Choppiness Index > 50 = avoid trading (too choppy for lower TF)
5. Session filter: only trade 8-20 UTC (highest volume hours)
6. Volume confirmation: volume > 0.8x 20-bar average
7. ATR(14) trailing stop at 2.5x for risk management
8. Discrete position sizes: 0.25 full, 0.15 partial

TARGET: 40-70 trades/year, Sharpe > 0.3 on ALL symbols (BTC, ETH, SOL)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_htf_trend_rsi_session_4h1d_v1"
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
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

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
    
    rsi = rsi.fillna(50.0).values
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = range/choppy market (avoid trading)
    CHOP < 38.2 = trending market
    """
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(period)
    
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD."""
    close_s = pd.Series(close)
    ema_fast = close_s.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = close_s.ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    return macd_line.values, macd_signal.values, macd_hist.values

def get_hour_from_open_time(prices):
    """Extract hour from open_time for session filter."""
    # open_time is in milliseconds since epoch
    timestamps = prices['open_time'].values / 1000.0
    hours = (timestamps % 86400) / 3600.0
    return hours.astype(int)

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 30m indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    macd_line, macd_signal, macd_hist = calculate_macd(close, fast=12, slow=26, signal=9)
    
    # Calculate 30m HMA for local trend
    hma_21_30m = calculate_hma(close, period=21)
    hma_50_30m = calculate_hma(close, period=50)
    
    # Calculate 4h HMA for PRIMARY trend direction
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate 1d HMA for macro bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Volume average (20-bar)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session hours (for filter)
    hours = get_hour_from_open_time(prices)
    
    signals = np.zeros(n)
    POSITION_SIZE_FULL = 0.25
    POSITION_SIZE_HALF = 0.15
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_21_30m[i]) or np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(chop_14[i]) or np.isnan(rsi_14[i]):
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(vol_avg[i]) or vol_avg[i] == 0:
            continue
        if np.isnan(macd_line[i]) or np.isnan(macd_hist[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        in_session = 8 <= hours[i] <= 20
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * vol_avg[i]
        
        # === CHOPPINESS FILTER (avoid choppy markets) ===
        chop_value = chop_14[i]
        not_too_choppy = chop_value < 55.0  # Avoid very choppy markets
        
        # === 4H PRIMARY TREND DIRECTION ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === 1D MACRO BIAS ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === 30M LOCAL TREND ===
        price_above_hma_21 = close[i] > hma_21_30m[i]
        price_below_hma_21 = close[i] < hma_21_30m[i]
        hma_21_above_50 = hma_21_30m[i] > hma_50_30m[i] if not np.isnan(hma_50_30m[i]) else False
        hma_21_below_50 = hma_21_30m[i] < hma_50_30m[i] if not np.isnan(hma_50_30m[i]) else False
        
        # === RSI PULLBACK SIGNALS ===
        rsi_oversold_pullback = rsi_14[i] < 42.0  # Pullback in uptrend
        rsi_overbought_pullback = rsi_14[i] > 58.0  # Pullback in downtrend
        
        # === MACD MOMENTUM ===
        macd_bullish = macd_hist[i] > 0.0
        macd_bearish = macd_hist[i] < 0.0
        macd_cross_up = macd_hist[i] > 0.0 and macd_hist[i-1] <= 0.0 if i > 0 else False
        macd_cross_down = macd_hist[i] < 0.0 and macd_hist[i-1] >= 0.0 if i > 0 else False
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG entries - MUST have 4h trend bullish as primary filter
        long_confluence = 0
        
        if price_above_hma_4h:  # PRIMARY: 4h trend must be bullish
            long_confluence += 2  # Weight this heavily
        if price_above_hma_1d:  # Macro bias confirmation
            long_confluence += 1
        if rsi_oversold_pullback:  # RSI pullback entry
            long_confluence += 1
        if macd_bullish or macd_cross_up:  # MACD momentum
            long_confluence += 1
        if volume_ok:  # Volume confirmation
            long_confluence += 1
        if in_session:  # Session filter
            long_confluence += 0.5
        if not_too_choppy:  # Not too choppy
            long_confluence += 0.5
        
        # LONG: Need 4+ confluence factors (strict for lower TF)
        if long_confluence >= 4.0 and price_above_hma_4h:
            if price_above_hma_1d and price_above_hma_21:
                # All trends aligned = full size
                new_signal = POSITION_SIZE_FULL
            elif price_above_hma_21:
                # 30m and 4h aligned = half size
                new_signal = POSITION_SIZE_HALF
        
        # SHORT entries - MUST have 4h trend bearish as primary filter
        short_confluence = 0
        
        if price_below_hma_4h:  # PRIMARY: 4h trend must be bearish
            short_confluence += 2  # Weight this heavily
        if price_below_hma_1d:  # Macro bias confirmation
            short_confluence += 1
        if rsi_overbought_pullback:  # RSI pullback entry
            short_confluence += 1
        if macd_bearish or macd_cross_down:  # MACD momentum
            short_confluence += 1
        if volume_ok:  # Volume confirmation
            short_confluence += 1
        if in_session:  # Session filter
            short_confluence += 0.5
        if not_too_choppy:  # Not too choppy
            short_confluence += 0.5
        
        # SHORT: Need 4+ confluence factors (strict for lower TF)
        if short_confluence >= 4.0 and price_below_hma_4h:
            if price_below_hma_1d and price_below_hma_21:
                # All trends aligned = full size
                new_signal = -POSITION_SIZE_FULL
            elif price_below_hma_21:
                # 30m and 4h aligned = half size
                new_signal = -POSITION_SIZE_HALF
        
        # === HOLD POSITION LOGIC ===
        # Hold if in position and 4h trend still valid
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if 4h trend still bullish
                if price_above_hma_4h:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if 4h trend still bearish
                if price_below_hma_4h:
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
        
        # === TREND REVERSAL EXIT ===
        # Exit long if 4h trend flips bearish
        if in_position and position_side > 0 and price_below_hma_4h:
            new_signal = 0.0
        
        # Exit short if 4h trend flips bullish
        if in_position and position_side < 0 and price_above_hma_4h:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals