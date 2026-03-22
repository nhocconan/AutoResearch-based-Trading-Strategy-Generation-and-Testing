#!/usr/bin/env python3
"""
Experiment #608: 30m Primary + 4h/1d HTF — KAMA Trend + Choppiness Regime + RSI Pullback

Hypothesis: 30m can work IF we use DOUBLE HTF filter (4h + 1d) for direction, 
and 30m ONLY for entry timing. This follows the proven pattern from #604 
(4h KAMA+CHOP+RSI, Sharpe=0.378) but adapted for 30m with stricter filters.

Key design choices:
1. 4h KAMA slope + 1d KAMA slope MUST agree (double HTF confirmation)
2. 30m RSI pullback within HTF trend (35-50 for longs, 50-65 for shorts)
3. Choppiness Index regime filter (trend-follow when CHOP<45)
4. Session filter (8-20 UTC) to avoid Asian session whipsaw
5. Volume filter (>0.8x 20-bar avg) to confirm participation
6. Conservative size (0.22) for lower TF fee sensitivity
7. 2.5*ATR trailing stop

Why this might work when #598 (30m) failed with Sharpe=-2.088:
- #598 used triple TF with too many conflicting filters
- This uses simpler logic: HTF direction + LTF pullback entry
- Session + volume filters reduce trade count to target 30-80/year
- Asymmetric RSI (longs 35-50, shorts 50-65) matches crypto behavior

Position sizing: 0.22 discrete (per Rule 4, max 0.40, lower for 30m)
Target: 30-80 trades/year on 30m (per Rule 10)
Stoploss: 2.5*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_kama_chop_rsi_4h1d_v1"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    Adapts smoothing based on market efficiency (trend vs noise).
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Price change over ER period
    price_change = np.abs(close_s.diff(er_period).values)
    
    # Sum of absolute price changes (volatility)
    volatility = np.zeros(n)
    for i in range(er_period, n):
        volatility[i] = np.sum(np.abs(np.diff(close[i-er_period:i+1])))
    
    # Efficiency Ratio (ER)
    with np.errstate(divide='ignore', invalid='ignore'):
        er = price_change / (volatility + 1e-10)
    er = np.nan_to_num(er, nan=0.0)
    er = np.clip(er, 0.0, 1.0)
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    # Adaptive smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    kama[:er_period] = np.nan
    
    return kama

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/range market (mean reversion works)
    CHOP < 38.2 = trending market (trend following works)
    """
    n = period
    atr_vals = calculate_atr(high, low, close, 14)
    
    atr_sum = pd.Series(atr_vals).rolling(window=n, min_periods=n).sum().values
    highest_high = pd.Series(high).rolling(window=n, min_periods=n).max().values
    lowest_low = pd.Series(low).rolling(window=n, min_periods=n).min().values
    price_range = highest_high - lowest_low
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(n)
    
    chop = np.clip(chop, 0.0, 100.0)
    chop = np.nan_to_num(chop, nan=50.0)
    
    return chop

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    return (open_time // (1000 * 3600)) % 24

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
    
    # Calculate 4h KAMA for intermediate trend
    kama_4h = calculate_kama(df_4h['close'].values, er_period=10, fast_period=2, slow_period=30)
    kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h)
    
    # Calculate 1d KAMA for primary trend
    kama_1d = calculate_kama(df_1d['close'].values, er_period=10, fast_period=2, slow_period=30)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Calculate 30m indicators
    kama_30m = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    chop_14 = calculate_choppiness(high, low, close, 14)
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    
    # Volume average (20 bars)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40, lower for 30m)
    POSITION_SIZE = 0.22
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(kama_30m[i]) or np.isnan(kama_4h_aligned[i]) or np.isnan(kama_1d_aligned[i]):
            continue
        if np.isnan(chop_14[i]) or np.isnan(atr_14[i]) or np.isnan(rsi_14[i]):
            continue
        if np.isnan(vol_avg_20[i]) or atr_14[i] == 0 or vol_avg_20[i] == 0:
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        utc_hour = get_utc_hour(open_time[i])
        in_session = 8 <= utc_hour <= 20
        
        # === VOLUME FILTER (>0.8x 20-bar average) ===
        volume_ratio = volume[i] / (vol_avg_20[i] + 1e-10)
        volume_ok = volume_ratio > 0.8
        
        # === 1D TREND BIAS (KAMA slope over 3 bars) ===
        kama_1d_slope_bull = kama_1d_aligned[i] > kama_1d_aligned[i-3] if i >= 3 else False
        kama_1d_slope_bear = kama_1d_aligned[i] < kama_1d_aligned[i-3] if i >= 3 else False
        
        # Price relative to 1d KAMA
        price_above_kama_1d = close[i] > kama_1d_aligned[i]
        price_below_kama_1d = close[i] < kama_1d_aligned[i]
        
        # === 4H TREND CONFIRMATION (KAMA slope over 2 bars) ===
        kama_4h_slope_bull = kama_4h_aligned[i] > kama_4h_aligned[i-2] if i >= 2 else False
        kama_4h_slope_bear = kama_4h_aligned[i] < kama_4h_aligned[i-2] if i >= 2 else False
        
        # Price relative to 4h KAMA
        price_above_kama_4h = close[i] > kama_4h_aligned[i]
        price_below_kama_4h = close[i] < kama_4h_aligned[i]
        
        # === 30M KAMA SLOPE (2 bars) ===
        kama_30m_slope_bull = kama_30m[i] > kama_30m[i-2] if i >= 2 else False
        kama_30m_slope_bear = kama_30m[i] < kama_30m[i-2] if i >= 2 else False
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_trend_regime = chop_14[i] < 45.0
        is_chop_regime = chop_14[i] > 55.0
        
        # === DOUBLE HTF TREND AGREEMENT ===
        htf_bull = kama_1d_slope_bull and kama_4h_slope_bull and price_above_kama_1d and price_above_kama_4h
        htf_bear = kama_1d_slope_bear and kama_4h_slope_bear and price_below_kama_1d and price_below_kama_4h
        
        # === ENTRY SIGNAL CALCULATION ===
        entry_signal = 0.0
        
        # --- TREND REGIME: Follow HTF trend with 30m pullback entries ---
        if is_trend_regime:
            # LONG: All HTF bull + 30m bull + RSI pullback (35-50) + session + volume
            if htf_bull and kama_30m_slope_bull and in_session and volume_ok:
                if 35.0 <= rsi_14[i] <= 50.0:
                    entry_signal = POSITION_SIZE
            
            # SHORT: All HTF bear + 30m bear + RSI bounce (50-65) + session + volume
            elif htf_bear and kama_30m_slope_bear and in_session and volume_ok:
                if 50.0 <= rsi_14[i] <= 65.0:
                    entry_signal = -POSITION_SIZE
        
        # --- CHOP REGIME: Mean reversion at RSI extremes ---
        elif is_chop_regime:
            # LONG: RSI < 35 + session + volume (no HTF filter needed in chop)
            if rsi_14[i] < 35.0 and in_session and volume_ok:
                entry_signal = POSITION_SIZE
            
            # SHORT: RSI > 65 + session + volume
            elif rsi_14[i] > 65.0 and in_session and volume_ok:
                entry_signal = -POSITION_SIZE
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        # === EXIT ON HTF TREND FLIP ===
        trend_flip_exit = False
        if in_position and position_side > 0:
            if kama_1d_slope_bear and price_below_kama_1d:
                trend_flip_exit = True
        
        if in_position and position_side < 0:
            if kama_1d_slope_bull and price_above_kama_1d:
                trend_flip_exit = True
        
        # === DETERMINE FINAL SIGNAL ===
        if stoploss_triggered or trend_flip_exit:
            new_signal = 0.0
        elif entry_signal != 0.0:
            new_signal = entry_signal
        elif in_position:
            new_signal = signals[i-1] if i > 0 else 0.0
        else:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = new_signal
    
    return signals