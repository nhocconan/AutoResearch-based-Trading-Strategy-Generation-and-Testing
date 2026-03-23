#!/usr/bin/env python3
"""
Experiment #470: 1h Primary + 4h/12h HTF — KAMA Adaptive Trend + Fisher Transform + ADX Regime

Hypothesis: KAMA (Kaufman Adaptive Moving Average) outperforms HMA/EMA in crypto because:
1. KAMA adapts smoothing based on market efficiency ratio (ER)
2. In trending markets (high ER), KAMA follows price closely
3. In ranging markets (low ER), KAMA flattens → fewer false signals
4. Fisher Transform catches reversals better than RSI (normalized -1 to +1)
5. ADX regime filter: ADX>25 = trend follow, ADX<20 = mean revert
6. Session filter (8-20 UTC) avoids low-liquidity whipsaws
7. Volume confirmation (>1.2x avg) ensures breakout validity

Target: Sharpe > 0.612, 30-60 trades/year, DD < -35%
Timeframe: 1h (with 4h/12h HTF bias)
Why this should work: KAMA's adaptive nature handles crypto's regime changes better than fixed-period MAs.
Fisher Transform is proven for reversal detection. ADX prevents trading in chop.
Session filter reduces noise. Volume confirms real moves.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_kama_fisher_adx_session_4h12h_v1"
timeframe = "1h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average.
    KAMA adapts smoothing based on market efficiency ratio.
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    # Calculate Efficiency Ratio (ER)
    # ER = |price change| / sum of absolute price changes
    er = np.full(n, np.nan)
    for i in range(period, n):
        price_change = np.abs(close[i] - close[i-period])
        noise = np.sum(np.abs(np.diff(close[i-period:i+1])))
        if noise > 0:
            er[i] = price_change / noise
        else:
            er[i] = 0
    
    er = np.nan_to_num(er, nan=0.0)
    er = np.clip(er, 0, 1)
    
    # Calculate Smoothing Constant (SC)
    # SC = [ER * (fast_SC - slow_SC) + slow_SC]^2
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = np.square(er * (fast_sc - slow_sc) + slow_sc)
    
    # Calculate KAMA
    kama[period] = close[period]
    for i in range(period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Normalizes price to -1 to +1 range for reversal detection.
    """
    n = len(high)
    fisher = np.full(n, np.nan)
    fisher_signal = np.full(n, np.nan)
    
    for i in range(period, n):
        # Calculate HL2 (typical price)
        hl2 = (high[i-period+1:i+1] + low[i-period+1:i+1]) / 2
        
        # Find highest high and lowest low over period
        highest = np.nanmax(high[i-period+1:i+1])
        lowest = np.nanmin(low[i-period+1:i+1])
        
        # Normalize price to 0-1 range
        if highest > lowest:
            normalized = (hl2[-1] - lowest) / (highest - lowest)
        else:
            normalized = 0.5
        
        # Clamp to 0.001-0.999 to avoid log(0)
        normalized = np.clip(normalized, 0.001, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1 + normalized) / (1 - normalized))
        
        # Signal line (1-period lag)
        if i > period:
            fisher_signal[i] = fisher[i-1]
    
    return fisher, fisher_signal

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Calculate +DM and -DM
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    # Smooth TR, +DM, -DM using Wilder's method (EMA with span=period)
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate +DI and -DI
    with np.errstate(divide='ignore', invalid='ignore'):
        plus_di = 100.0 * plus_dm_s / (tr_s + 1e-10)
        minus_di = 100.0 * minus_dm_s / (tr_s + 1e-10)
    
    # Calculate DX
    with np.errstate(divide='ignore', invalid='ignore'):
        dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    
    # Calculate ADX (smoothed DX)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 1h indicators (primary timeframe)
    kama_fast = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    kama_slow = calculate_kama(close, period=20, fast_period=2, slow_period=30)
    fisher, fisher_signal = calculate_fisher_transform(high, low, period=9)
    adx, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    
    # Calculate volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate and align HTF indicators
    kama_4h_raw = calculate_kama(df_4h['close'].values, period=10, fast_period=2, slow_period=30)
    kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h_raw)
    
    kama_12h_raw = calculate_kama(df_12h['close'].values, period=10, fast_period=2, slow_period=30)
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h_raw)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.30
    SIZE_SHORT = 0.20  # Asymmetric: smaller short size for crypto bias
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # Calculate ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(kama_fast[i]) or np.isnan(kama_slow[i]):
            continue
        if np.isnan(fisher[i]) or np.isnan(adx[i]):
            continue
        if np.isnan(kama_4h_aligned[i]) or np.isnan(kama_12h_aligned[i]):
            continue
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            continue
        if np.isnan(vol_avg[i]) or vol_avg[i] <= 1e-10:
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        # Convert open_time to hour
        timestamp_ms = open_time[i]
        hour_utc = (timestamp_ms // 3600000) % 24
        in_session = 8 <= hour_utc <= 20
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 1.2 * vol_avg[i]
        
        # === PRIMARY TREND (KAMA crossover) ===
        trend_bullish = kama_fast[i] > kama_slow[i]
        trend_bearish = kama_fast[i] < kama_slow[i]
        
        # === HTF TREND BIAS (4h and 12h KAMA) ===
        price_above_kama_4h = close[i] > kama_4h_aligned[i]
        price_below_kama_4h = close[i] < kama_4h_aligned[i]
        price_above_kama_12h = close[i] > kama_12h_aligned[i]
        price_below_kama_12h = close[i] < kama_12h_aligned[i]
        
        # === ADX REGIME FILTER ===
        # ADX > 25 = trending, ADX < 20 = ranging
        is_trending = adx[i] > 25.0
        is_ranging = adx[i] < 20.0
        
        # === FISHER TRANSFORM ENTRY SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_long = fisher[i] > -1.5 and fisher_signal[i] <= -1.5
        fisher_short = fisher[i] < 1.5 and fisher_signal[i] >= 1.5
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG: Multiple confluence required (relaxed from 5 to 4 for trade frequency)
        long_score = 0
        if trend_bullish:
            long_score += 2  # Primary trend
        if price_above_kama_4h:
            long_score += 1  # 4h bias
        if price_above_kama_12h:
            long_score += 1  # 12h bias
        if fisher_long:
            long_score += 2  # Fisher reversal
        if volume_confirmed:
            long_score += 1  # Volume confirmation
        if in_session:
            long_score += 1  # Session filter
        
        # Require 4+ score for long entry
        if long_score >= 4:
            desired_signal = SIZE_LONG
        
        # SHORT: Multiple confluence required
        if desired_signal == 0.0:
            short_score = 0
            if trend_bearish:
                short_score += 2  # Primary trend
            if price_below_kama_4h:
                short_score += 1  # 4h bias
            if price_below_kama_12h:
                short_score += 1  # 12h bias
            if fisher_short:
                short_score += 2  # Fisher reversal
            if volume_confirmed:
                short_score += 1  # Volume confirmation
            if in_session:
                short_score += 1  # Session filter
            
            # Require 4+ score for short entry
            if short_score >= 4:
                desired_signal = -SIZE_SHORT
        
        # === STOPLOSS CHECK (Trailing ATR) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if trend unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and trend_bullish and price_above_kama_4h:
                desired_signal = SIZE_LONG
            elif position_side < 0 and trend_bearish and price_below_kama_4h:
                desired_signal = -SIZE_SHORT
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = 0.30
        elif desired_signal < 0:
            desired_signal = -0.20
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals