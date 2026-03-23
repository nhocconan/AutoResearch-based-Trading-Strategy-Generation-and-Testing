#!/usr/bin/env python3
"""
Experiment #015: 1h Primary + 4h/1d HTF — Regime-Adaptive with Session/Volume Filter

Hypothesis: Previous failures show that single-regime strategies fail because crypto
alternates between trending and ranging. This strategy ADAPTS to regime using Choppiness
Index, but with LOOSER thresholds than failed attempts (#006, #008, #010).

Key differences from failed CRSI/Chop strategies:
1. CHOP thresholds: >55 range, <45 trend (not 61.8/38.2 which are too strict)
2. RSI(14) instead of CRSI (CRSI failed in #006, #008, #010)
3. Session filter: only 8-20 UTC (high liquidity, reduces false breakouts)
4. Volume confirmation: >0.8x 20-bar average (filters low-liquidity traps)
5. Asymmetric sizing: 0.25 in trend, 0.20 in range (conservative in chop)
6. 4h HMA for trend direction (proven in #007, #009)
7. 1d HMA for regime confirmation (avoid counter-trend in strong trends)

Why this might work:
- Regime-adaptive: different logic for trend vs range (research-backed)
- Session filter: avoids Asian session whipsaws (8-20 UTC = EU/US overlap)
- Volume filter: confirms genuine moves vs fakeouts
- HTF trend alignment: 4h HMA + 1d HMA confluence
- LOOSE enough to generate trades (RSI 35-65 zone, not extremes)

Entry conditions (designed for 40-80 trades/year on 1h):
- Trend regime (CHOP<45): RSI pullback to 40-50 (long) or 50-60 (short) + HTF trend
- Range regime (CHOP>55): RSI extremes <35 (long) or >65 (short) + BB mean reversion
- Volume > 0.8x avg, session 8-20 UTC

Stoploss: 2.5*ATR trailing, signal→0 when hit
Position size: 0.25 (trend), 0.20 (range) — discrete levels per Rule 4
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_adaptive_session_volume_4h1d_v2"
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

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = range/choppy, CHOP < 38.2 = trending
    We use looser thresholds: >55 = range, <45 = trend
    """
    n = period
    atr_sum = np.zeros(len(close))
    hh = np.zeros(len(close))
    ll = np.zeros(len(close))
    
    for i in range(n, len(close)):
        # Sum of ATR over period
        atr_window = []
        for j in range(i - n + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_window.append(tr)
        atr_sum[i] = sum(atr_window)
        
        # Highest high and lowest low over period
        hh[i] = np.max(high[i - n + 1:i + 1])
        ll[i] = np.min(low[i - n + 1:i + 1])
    
    # Avoid division by zero
    range_hl = hh - ll
    range_hl[range_hl == 0] = 1e-10
    
    chop = 100.0 * np.log10(atr_sum / range_hl + 1e-10) / np.log10(n + 1e-10)
    chop[:n] = np.nan
    
    return chop

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    
    # %B (position within bands)
    pct_b = (close - lower) / (upper - lower + 1e-10)
    
    return upper.values, lower.values, pct_b.values

def calculate_volume_avg(volume, period=20):
    """Calculate rolling average volume."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_avg

def get_hour_from_open_time(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
    return (open_time // 3600000) % 24

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
    
    # Calculate 1d HMA for regime confirmation
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    chop_14 = calculate_choppiness_index(high, low, close, period=14)
    bb_upper, bb_lower, bb_pct_b = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    vol_avg = calculate_volume_avg(volume, period=20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    SIZE_TREND = 0.25  # More confident in trend regime
    SIZE_RANGE = 0.20  # More conservative in choppy regime
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(rsi_14[i]) or np.isnan(chop_14[i]):
            continue
        if np.isnan(bb_pct_b[i]) or np.isnan(vol_avg[i]):
            continue
        if atr_14[i] == 0 or vol_avg[i] == 0:
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        hour = get_hour_from_open_time(open_time[i])
        in_session = 8 <= hour <= 20
        
        # === VOLUME FILTER ===
        volume_confirmed = volume[i] > 0.8 * vol_avg[i]
        
        # === REGIME DETECTION ===
        chop = chop_14[i]
        is_trend_regime = chop < 45.0
        is_range_regime = chop > 55.0
        # Between 45-55 = transition, no new entries
        
        # === 4H TREND DIRECTION ===
        hma_4h_slope_bull = hma_4h_aligned[i] > hma_4h_aligned[i-3] if i >= 3 else False
        hma_4h_slope_bear = hma_4h_aligned[i] < hma_4h_aligned[i-3] if i >= 3 else False
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === 1D REGIME CONFIRMATION ===
        hma_1d_slope_bull = hma_1d_aligned[i] > hma_1d_aligned[i-2] if i >= 2 else False
        hma_1d_slope_bear = hma_1d_aligned[i] < hma_1d_aligned[i-2] if i >= 2 else False
        
        # === REGIME-ADAPTIVE ENTRY LOGIC ===
        new_signal = 0.0
        current_size = SIZE_TREND if is_trend_regime else SIZE_RANGE
        
        # --- TREND REGIME (CHOP < 45) ---
        if is_trend_regime:
            # Long: RSI pullback to 40-50 zone + 4h bullish + volume confirmed
            rsi_pullback_long = 40 <= rsi_14[i] <= 50
            trend_bullish = hma_4h_slope_bull and price_above_hma_4h
            daily_confirms = not hma_1d_slope_bear  # 1d not strongly bearish
            
            if rsi_pullback_long and trend_bullish and daily_confirms and in_session and volume_confirmed:
                new_signal = current_size
            
            # Short: RSI pullback to 50-60 zone + 4h bearish + volume confirmed
            rsi_pullback_short = 50 <= rsi_14[i] <= 60
            trend_bearish = hma_4h_slope_bear and price_below_hma_4h
            daily_confirms_short = not hma_1d_slope_bull  # 1d not strongly bullish
            
            if rsi_pullback_short and trend_bearish and daily_confirms_short and in_session and volume_confirmed:
                new_signal = -current_size
        
        # --- RANGE REGIME (CHOP > 55) ---
        elif is_range_regime:
            # Long: RSI < 35 + BB %B < 0.2 (oversold mean reversion)
            rsi_oversold = rsi_14[i] < 35
            bb_oversold = bb_pct_b[i] < 0.2
            
            # Only if not in strong downtrend (1d not bearish)
            range_ok_long = not hma_1d_slope_bear
            
            if rsi_oversold and bb_oversold and range_ok_long and in_session and volume_confirmed:
                new_signal = current_size
            
            # Short: RSI > 65 + BB %B > 0.8 (overbought mean reversion)
            rsi_overbought = rsi_14[i] > 65
            bb_overbought = bb_pct_b[i] > 0.8
            
            # Only if not in strong uptrend (1d not bullish)
            range_ok_short = not hma_1d_slope_bull
            
            if rsi_overbought and bb_overbought and range_ok_short and in_session and volume_confirmed:
                new_signal = -current_size
        
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
        
        # === EXIT ON TREND FLIP ===
        if in_position and position_side > 0:
            if hma_4h_slope_bear and price_below_hma_4h:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if hma_4h_slope_bull and price_above_hma_4h:
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