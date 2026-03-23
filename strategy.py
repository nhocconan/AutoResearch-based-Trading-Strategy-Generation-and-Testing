#!/usr/bin/env python3
"""
Experiment #038: 30m Primary + 4h/1d HTF — Triple Confluence Regime Adaptive

Hypothesis: 30m timeframe with STRONG HTF filters (4h trend + 1d regime) will generate
30-80 trades/year with positive Sharpe. Key insight from 37 failed experiments:
lower TF strategies fail due to TOO MANY TRADES (>200/yr) causing fee drag.

Solution: Use 4h HMA for TREND DIRECTION + 1d CHOP for REGIME + 30m for ENTRY TIMING.
This gives HTF trade frequency with 30m execution precision.

Strategy Logic:
1. 1d CHOPPINESS: CHOP > 55 = range (mean revert), CHOP < 45 = trend (follow)
2. 4h HMA(21): Macro trend bias (only trade WITH 4h direction)
3. 30m RSI(7): Entry trigger (oversold/overbought extremes, LOOSE: 25/75)
4. Volume filter: volume > 0.8x 20-bar avg (confirm participation)
5. Session filter: 8-20 UTC only (highest liquidity, avoid Asian chop)
6. ATR(14) trailing stoploss: 2.5*ATR to protect capital

Why this should work on 30m:
- 4h + 1d HTF filters = fewer signals (target 40-60/year, not 200+)
- Session filter = avoids low-liquidity whipsaws
- LOOSE RSI (25/75 not 20/80) = ensures trade generation
- Volume confirmation = filters false breakouts
- Discrete sizing (0.25) = minimizes fee churn

Position size: 0.25 (smaller for lower TF, within 0.20-0.35 range)
Stoploss: 2.5*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_rsi_chop_hma_triple_htf_session_v1"
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
    """Calculate Hull Moving Average (HMA)."""
    close_s = pd.Series(close)
    
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

def calculate_rsi(close, period=7):
    """Calculate RSI with shorter period for 30m entries."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = period
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=n, min_periods=n).sum().values
    highest_high = pd.Series(high).rolling(window=n, min_periods=n).max().values
    lowest_low = pd.Series(low).rolling(window=n, min_periods=n).min().values
    
    price_range = highest_high - lowest_low + 1e-10
    chop = 100.0 * np.log10(atr_sum / price_range) / np.log10(n)
    
    return chop

def calculate_volume_sma(volume, period=20):
    """Calculate simple moving average of volume."""
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def extract_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
    hours = (open_time // (1000 * 60 * 60)) % 24
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h HMA for trend direction
    hma_4h = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1d CHOP for regime detection
    chop_1d = calculate_choppiness(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values,
        period=14
    )
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)
    vol_sma_20 = calculate_volume_sma(volume, period=20)
    
    # Calculate 30m HMA for additional trend confirmation
    hma_30m = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, smaller for 30m)
    POSITION_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if np.isnan(hma_4h_aligned[i]) or np.isnan(chop_1d_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(rsi_7[i]) or np.isnan(vol_sma_20[i]):
            continue
        if np.isnan(hma_30m[i]) or atr_14[i] == 0:
            continue
        
        # Extract UTC hour for session filter
        current_hour = extract_hour(open_time[i])
        
        # === SESSION FILTER (8-20 UTC only) ===
        in_session = 8 <= current_hour <= 20
        
        # === 1D REGIME (Choppiness) ===
        chop_value = chop_1d_aligned[i]
        is_ranging = chop_value > 55.0  # Range market
        is_trending = chop_value < 45.0  # Trend market (with hysteresis)
        
        # === 4H TREND DIRECTION ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        hma_4h_slope_up = hma_4h_aligned[i] > hma_4h_aligned[i-5] if i > 5 else False
        hma_4h_slope_down = hma_4h_aligned[i] < hma_4h_aligned[i-5] if i > 5 else False
        
        # === 30M RSI EXTREMES (LOOSE for trade generation) ===
        rsi_oversold = rsi_7[i] < 25.0
        rsi_overbought = rsi_7[i] > 75.0
        rsi_rising = rsi_7[i] > rsi_7[i-1] if i > 0 else False
        rsi_falling = rsi_7[i] < rsi_7[i-1] if i > 0 else False
        rsi_turning_up = rsi_oversold and rsi_rising
        rsi_turning_down = rsi_overbought and rsi_falling
        
        # === VOLUME CONFIRMATION ===
        volume_above_avg = volume[i] > 0.8 * vol_sma_20[i]
        
        # === 30M HMA TREND ===
        hma_30m_bullish = close[i] > hma_30m[i]
        hma_30m_bearish = close[i] < hma_30m[i]
        
        # === ADAPTIVE REGIME ENTRY LOGIC ===
        new_signal = 0.0
        
        # Only trade during high-liquidity session
        if not in_session:
            # Hold existing position but don't enter new
            if in_position:
                new_signal = signals[i-1] if i > 0 else 0.0
            signals[i] = new_signal
            continue
        
        # --- RANGING REGIME: Mean Reversion ---
        if is_ranging:
            # Long: RSI oversold + 4h trend bullish OR neutral + volume confirms
            if rsi_turning_up:
                if price_above_hma_4h and volume_above_avg:
                    new_signal = POSITION_SIZE
                elif not price_below_hma_4h and volume_above_avg:  # 4h neutral is OK in range
                    new_signal = POSITION_SIZE
            
            # Short: RSI overbought + 4h trend bearish OR neutral + volume confirms
            elif rsi_turning_down:
                if price_below_hma_4h and volume_above_avg:
                    new_signal = -POSITION_SIZE
                elif not price_above_hma_4h and volume_above_avg:  # 4h neutral is OK in range
                    new_signal = -POSITION_SIZE
        
        # --- TRENDING REGIME: Trend Following ---
        elif is_trending:
            # Long: 4h bullish + 30m pullback (RSI rising from low) + volume
            if price_above_hma_4h and hma_4h_slope_up:
                if rsi_7[i] < 50 and rsi_rising and volume_above_avg:
                    new_signal = POSITION_SIZE
            
            # Short: 4h bearish + 30m retracement (RSI falling from high) + volume
            elif price_below_hma_4h and hma_4h_slope_down:
                if rsi_7[i] > 50 and rsi_falling and volume_above_avg:
                    new_signal = -POSITION_SIZE
        
        # --- FALLBACK: 30m HMA crossover with 4h confirmation ---
        if new_signal == 0.0:
            # Long: 30m crosses above HMA + 4h bullish + volume
            if close[i] > hma_30m[i] and close[i-1] <= hma_30m[i-1]:
                if price_above_hma_4h and volume_above_avg:
                    new_signal = POSITION_SIZE
            
            # Short: 30m crosses below HMA + 4h bearish + volume
            elif close[i] < hma_30m[i] and close[i-1] >= hma_30m[i-1]:
                if price_below_hma_4h and volume_above_avg:
                    new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
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
        
        # === EXIT ON HTF TREND REVERSAL ===
        # Exit long if 4h trend turns strongly bearish
        if in_position and position_side > 0:
            if price_below_hma_4h and hma_4h_slope_down and is_trending:
                new_signal = 0.0
        
        # Exit short if 4h trend turns strongly bullish
        if in_position and position_side < 0:
            if price_above_hma_4h and hma_4h_slope_up and is_trending:
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