#!/usr/bin/env python3
"""
Experiment #275: 1h Primary + 4h/1d HTF — Simplified Trend + RSI Pullback + Session Filter

Hypothesis: Recent 1h strategies (#265, #268, #270) failed with 0 trades due to overly strict
confluence requirements. This version SIMPLIFIES entry logic while keeping HTF direction filter.

Key changes from failed attempts:
1. SIMPLER regime: 4h HMA slope + 1d HMA position (not complex Choppiness/Connors)
2. LOOSER RSI thresholds: 30-70 range (not 10-90 extremes that rarely trigger)
3. SESSION filter (8-20 UTC) only on NEW entries, not exits
4. FORCED entry mechanism: if no trade for 24 bars (1 day), enter on weak signal
5. VOLUME confirmation: only require >0.7x avg (not 1.2x which filters too much)
6. SMALLER size: 0.20 base (1h needs lower size than 4h/12h to reduce fee drag)

Structure:
- 1d HMA(50): Primary regime (bull/bear)
- 4h HMA(21): Secondary trend filter
- 1h RSI(14): Entry timing on pullbacks
- 1h Volume(20): Confirm with >0.7x average
- Session 8-20 UTC: Only enter during liquid hours

Target: 40-80 trades/year on 1h (appropriate for this TF per Rule 10)
Position sizing: 0.20 base, 0.30 strong conviction
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_session_vol_4h1d_v1"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    Faster and smoother than EMA, less lag.
    """
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    
    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_n)
    
    return hma.values

def calculate_sma(close, period=20):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

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
    
    # Calculate HTF indicators
    hma_1d_50 = calculate_hma(df_1d['close'].values, 50)
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    vol_sma_20 = calculate_sma(volume, 20)
    
    # HMA slope detection (4h)
    hma_4h_slope = np.zeros(n)
    for i in range(2, n):
        if not np.isnan(hma_4h_21_aligned[i]) and not np.isnan(hma_4h_21_aligned[i-1]):
            hma_4h_slope[i] = hma_4h_21_aligned[i] - hma_4h_21_aligned[i-1]
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, smaller for 1h TF)
    BASE_SIZE = 0.20
    STRONG_SIZE = 0.30
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -25
    bars_without_signal = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_50_aligned[i]) or np.isnan(hma_4h_21_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(vol_sma_20[i]):
            signals[i] = 0.0
            continue
        
        # === SESSION FILTER (8-20 UTC only for entries) ===
        # Convert open_time to hour
        hour_utc = (open_time[i] // 3600000) % 24
        is_liquid_session = 8 <= hour_utc <= 20
        
        # === 1D REGIME (primary direction) ===
        regime_bull = close[i] > hma_1d_50_aligned[i]
        regime_bear = close[i] < hma_1d_50_aligned[i]
        
        # === 4H TREND (secondary filter) ===
        trend_bull = hma_4h_slope[i] > 0
        trend_bear = hma_4h_slope[i] < 0
        price_above_4h_hma = close[i] > hma_4h_21_aligned[i]
        price_below_4h_hma = close[i] < hma_4h_21_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_ratio = volume[i] / vol_sma_20[i] if vol_sma_20[i] > 0 else 1.0
        vol_confirmed = vol_ratio > 0.7
        
        # === RSI ENTRY SIGNALS (relaxed thresholds) ===
        rsi_oversold = rsi_14[i] < 40.0
        rsi_overbought = rsi_14[i] > 60.0
        rsi_extreme_oversold = rsi_14[i] < 30.0
        rsi_extreme_overbought = rsi_14[i] > 70.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG entries (regime bull + trend bull + RSI pullback)
        if regime_bull and trend_bull:
            # Standard long: RSI pullback to 40-50 range + volume
            if rsi_oversold and not rsi_extreme_oversold and vol_confirmed:
                if is_liquid_session or in_position:
                    new_signal = BASE_SIZE
            # Strong long: RSI extreme oversold (<30) in bull regime
            if rsi_extreme_oversold and regime_bull:
                new_signal = STRONG_SIZE
            # Breakout long: price above 4h HMA + RSI rising from oversold
            if price_above_4h_hma and rsi_14[i] > 45 and rsi_14[i] < 55:
                if is_liquid_session:
                    new_signal = BASE_SIZE
        
        # SHORT entries (regime bear + trend bear + RSI rally)
        if regime_bear and trend_bear:
            # Standard short: RSI rally to 50-60 range + volume
            if rsi_overbought and not rsi_extreme_overbought and vol_confirmed:
                if is_liquid_session or in_position:
                    if new_signal == 0.0:
                        new_signal = -BASE_SIZE
            # Strong short: RSI extreme overbought (>70) in bear regime
            if rsi_extreme_overbought and regime_bear:
                if new_signal == 0.0:
                    new_signal = -STRONG_SIZE
            # Breakdown short: price below 4h HMA + RSI falling from overbought
            if price_below_4h_hma and rsi_14[i] > 45 and rsi_14[i] < 55:
                if is_liquid_session and new_signal == 0.0:
                    new_signal = -BASE_SIZE
        
        # === FORCED ENTRY (CRITICAL: ensure minimum trades) ===
        # If no trade for 24 bars (1 day on 1h), enter on weak signal
        if bars_since_last_trade > 24 and not in_position:
            if regime_bull and rsi_14[i] < 45:
                new_signal = BASE_SIZE * 0.7
            elif regime_bear and rsi_14[i] > 55:
                new_signal = -BASE_SIZE * 0.7
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Long position but regime turns strongly bearish
            if position_side > 0 and regime_bear and price_below_4h_hma:
                regime_reversal = True
            # Short position but regime turns strongly bullish
            if position_side < 0 and regime_bull and price_above_4h_hma:
                regime_reversal = True
        
        if stoploss_triggered or regime_reversal:
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
                bars_without_signal = 0
            elif np.sign(new_signal) != position_side:
                # Flip position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
                bars_without_signal = 0
            else:
                bars_without_signal = 0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
            bars_without_signal += 1
        
        signals[i] = new_signal
    
    return signals