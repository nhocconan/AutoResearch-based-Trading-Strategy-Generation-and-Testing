#!/usr/bin/env python3
"""
Experiment #135: 1h Primary + 4h/1d HTF — Fisher Transform + ADX Trend + Session Filter

Hypothesis: Previous 1h strategies failed due to either excessive trading (fee drag) or 
overly restrictive entries (0 trades). This strategy uses:

1. 4h HMA(21) for TREND DIRECTION - only trade in HTF trend direction
2. 1h Fisher Transform(9) for ENTRY TIMING - catches reversals better than RSI
3. ADX(14) > 20 - confirms trend strength, avoids choppy whipsaws
4. Session Filter (8-20 UTC) - avoids low-liquidity Asian session noise
5. Volume Filter (>0.7x 20-bar avg) - confirms participation
6. 1d HMA slope - major trend bias filter

Why this should work:
- Fisher Transform has superior reversal detection in bear/range markets
- 4h trend filter prevents counter-trend trades that failed in 2022 crash
- Session + volume filters reduce false signals during low-liquidity periods
- Asymmetric sizing (0.25 long, 0.20 short) accounts for bear market bias
- Target: 40-70 trades/year (1h with strict filters = manageable fee drag)

Timeframe: 1h (REQUIRED)
HTF: 4h + 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.20-0.25 discrete
Stoploss: 2.0 * ATR(14) trailing
Target trades: 40-70/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_adx_session_4h1d_v1"
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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX using standard Wilder's method."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / np.where(atr > 0, atr, 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / np.where(atr > 0, atr, 1e-10)
    
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) > 0, (plus_di + minus_di), 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Fisher = 0.5 * ln((1 + X) / (1 - X)) where X = 2 * (close - LL) / (HH - LL) - 1
    """
    highest = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    x = 2 * (close - lowest) / np.where((highest - lowest) > 0, (highest - lowest), 1e-10) - 1
    x = np.clip(x, -0.999, 0.999)
    
    fisher = 0.5 * np.log((1 + x) / (1 - x))
    
    return fisher

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_hma_slope(hma_values, lookback=5):
    """Calculate HMA slope as percentage change."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        if hma_values[i - lookback] != 0:
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback] * 100
    return slope

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    return pd.to_datetime(open_time, unit='ms').dt.hour.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_slope = calculate_hma_slope(hma_4h_21, 5)
    
    # Calculate 1d indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_slope = calculate_hma_slope(hma_1d_21, 5)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_slope_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_slope)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    adx_14, plus_di, minus_di = calculate_adx(high, low, close, 14)
    fisher_9 = calculate_fisher_transform(high, low, close, 9)
    
    # Volume average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter (8-20 UTC)
    utc_hour = get_utc_hour(open_time)
    in_session = (utc_hour >= 8) & (utc_hour <= 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_SIZE = 0.25
    SHORT_SIZE = 0.20
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_4h_slope_aligned[i]):
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_slope_aligned[i]):
            continue
        
        if np.isnan(adx_14[i]) or np.isnan(fisher_9[i]):
            continue
        
        if np.isnan(vol_avg_20[i]):
            continue
        
        # === 4H TREND DIRECTION ===
        trend_4h_bullish = hma_4h_slope_aligned[i] > 0.15
        trend_4h_bearish = hma_4h_slope_aligned[i] < -0.15
        price_above_4h_hma = close[i] > hma_4h_21_aligned[i]
        price_below_4h_hma = close[i] < hma_4h_21_aligned[i]
        
        # === 1D MAJOR TREND BIAS ===
        trend_1d_bullish = hma_1d_slope_aligned[i] > 0.2
        trend_1d_bearish = hma_1d_slope_aligned[i] < -0.2
        
        # === ADX TREND STRENGTH ===
        strong_trend = adx_14[i] > 20
        weak_trend = adx_14[i] < 25
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_oversold = fisher_9[i] < -1.5
        fisher_overbought = fisher_9[i] > 1.5
        fisher_neutral = (fisher_9[i] > -1.0) and (fisher_9[i] < 1.0)
        
        # Fisher cross detection
        fisher_cross_up = False
        fisher_cross_down = False
        if i > 100:
            fisher_cross_up = (fisher_9[i] > -1.5) and (fisher_9[i-1] <= -1.5)
            fisher_cross_down = (fisher_9[i] < 1.5) and (fisher_9[i-1] >= 1.5)
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.7 * vol_avg_20[i]
        
        # === SESSION FILTER ===
        session_ok = in_session[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - require 4h bullish bias + 3+ confluence
        long_score = 0
        
        # 4h trend alignment (most important)
        if trend_4h_bullish or price_above_4h_hma:
            long_score += 2
        
        # Fisher signal
        if fisher_oversold or fisher_cross_up:
            long_score += 2
        
        # ADX confirmation
        if strong_trend or weak_trend:
            long_score += 1
        
        # Volume confirmation
        if volume_ok:
            long_score += 1
        
        # Session filter (optional bonus)
        if session_ok:
            long_score += 1
        
        # 1d trend confirmation
        if trend_1d_bullish or (hma_1d_slope_aligned[i] > -0.1):
            long_score += 1
        
        # Need 5+ score for long entry (relaxed from 4+ conditions)
        if long_score >= 5 and bars_since_last_trade > 12:
            new_signal = LONG_SIZE
        elif long_score >= 4 and bars_since_last_trade > 36:
            new_signal = LONG_SIZE * 0.6
        
        # SHORT ENTRIES - require 4h bearish bias + 3+ confluence
        short_score = 0
        
        # 4h trend alignment
        if trend_4h_bearish or price_below_4h_hma:
            short_score += 2
        
        # Fisher signal
        if fisher_overbought or fisher_cross_down:
            short_score += 2
        
        # ADX confirmation
        if strong_trend or weak_trend:
            short_score += 1
        
        # Volume confirmation
        if volume_ok:
            short_score += 1
        
        # Session filter
        if session_ok:
            short_score += 1
        
        # 1d trend confirmation
        if trend_1d_bearish or (hma_1d_slope_aligned[i] < 0.1):
            short_score += 1
        
        # Need 5+ score for short entry
        if short_score >= 5 and bars_since_last_trade > 12:
            new_signal = -SHORT_SIZE
        elif short_score >= 4 and bars_since_last_trade > 36:
            new_signal = -SHORT_SIZE * 0.6
        
        # === FREQUENCY SAFEGUARD - force trades if none for 100 bars ===
        if bars_since_last_trade > 100 and new_signal == 0.0 and not in_position:
            if trend_4h_bullish and fisher_9[i] < -0.5:
                new_signal = LONG_SIZE * 0.4
            elif trend_4h_bearish and fisher_9[i] > 0.5:
                new_signal = -SHORT_SIZE * 0.4
            elif fisher_9[i] < -1.2:
                new_signal = LONG_SIZE * 0.3
            elif fisher_9[i] > 1.2:
                new_signal = -SHORT_SIZE * 0.3
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.0 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.0 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and trend_4h_bearish and adx_14[i] > 25:
                trend_reversal = True
            if position_side < 0 and trend_4h_bullish and adx_14[i] > 25:
                trend_reversal = True
        
        # === FISHER EXTREME EXIT ===
        fisher_exit = False
        if in_position and position_side != 0:
            if position_side > 0 and fisher_9[i] > 2.0:
                fisher_exit = True
            if position_side < 0 and fisher_9[i] < -2.0:
                fisher_exit = True
        
        if stoploss_triggered or trend_reversal or fisher_exit:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals