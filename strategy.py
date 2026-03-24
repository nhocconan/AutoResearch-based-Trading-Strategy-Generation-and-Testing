#!/usr/bin/env python3
"""
Experiment #038: 30m Primary + 4h/1d HTF — RSI Pullback with Volume/Session Confluence

Hypothesis: Lower TF (30m) strategies fail due to either 0 trades (too strict) or 
too many trades (fee drag). This uses PROVEN pattern from best performers:
1. 4h HMA for TREND DIRECTION (not entry) - avoids counter-trend trades
2. 1d HMA for MAJOR REGIME - only trade with major trend for higher win rate
3. 30m RSI(7) pullback for ENTRY TIMING - enters on dips within uptrend
4. Volume filter (>1.2x 20-period avg) - confirms institutional participation
5. Session filter (8-20 UTC) - high liquidity hours, avoids Asian chop
6. Choppiness Index < 55 - avoid range-bound whipsaw

Key insight: 4h/1d tell you WHAT direction, 30m tells you WHEN to enter.
This gives HTF trade frequency (30-60/year) with 30m execution precision.

Entry Logic:
- LONG: 4h HMA bull + 1d HMA bull + RSI(7) 35-45 + volume>1.2x + CHOP<55 + 8-20 UTC
- SHORT: 4h HMA bear + 1d HMA bear + RSI(7) 55-65 + volume>1.2x + CHOP<55 + 8-20 UTC
- Size: 0.25 (conservative for 30m), discrete levels
- Stoploss: 2.5x ATR trailing

Why this should work:
- RSI(7) pullback is proven mean-reversion within trend (Larry Connors research)
- Volume filter eliminates fake breakouts
- Session filter reduces 40% of low-quality trades
- HTF alignment ensures we're not fighting major trend
- Expected: 40-70 trades/year, Sharpe>0.4

Risk: 2.5x ATR trailing stop, max signal 0.30, leverage=1.0
Target: Sharpe>0.4, trades>30/symbol train, >3/symbol test, DD>-40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_rsi_pullback_4h1d_vol_session_chop_v1"
timeframe = "30m"
leverage = 1.0

def calculate_rsi(close, period=7):
    """RSI with proper min_periods"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_hma(close, period=21):
    """Hull Moving Average for trend detection"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half_period = max(period // 2, 1)
    sqrt_period = max(int(np.sqrt(period)), 1)
    
    def wma(series, span):
        if len(series) < span:
            return np.full(len(series), np.nan)
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=float)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half_period)
    wma_full = wma(close, period)
    raw_hma = 2.0 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_period)
    
    return hma

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - regime detection (avoid choppy markets)"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        if highest_high == lowest_low:
            chop[i] = 100.0
            continue
        
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            prev_close = close[j-1] if j > 0 else close[j]
            tr = max(high[j] - low[j], abs(high[j] - prev_close), abs(low[j] - prev_close))
            atr_sum += tr
        
        chop[i] = 100.0 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_ratio(volume, period=20):
    """Volume ratio vs rolling average"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_ratio = np.full(n, np.nan)
    for i in range(period, n):
        avg_vol = np.mean(volume[i - period:i])
        if avg_vol > 1e-10:
            vol_ratio[i] = volume[i] / avg_vol
        else:
            vol_ratio[i] = 1.0
    
    return vol_ratio

def get_utc_hour(open_time):
    """Extract UTC hour from timestamp (milliseconds)"""
    # open_time is in milliseconds since epoch
    return (open_time // (1000 * 60 * 60)) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values if "open_time" in prices.columns else np.arange(len(close)) * 1800000  # 30m in ms
    n = len(close)
    
    # === LOAD HTF DATA ONCE BEFORE LOOP (Rule 1 - CRITICAL) ===
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # === CALCULATE AND ALIGN HTF INDICATORS ===
    # 4h HMA(21) for intermediate trend
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # 1d HMA(21) for major trend regime
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # === CALCULATE PRIMARY (30m) INDICATORS ===
    rsi_7 = calculate_rsi(close, period=7)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    REDUCED_SIZE = 0.15
    MAX_SIZE = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(200, n):
        # === SKIP IF INDICATORS NOT READY ===
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi_7[i]) or np.isnan(chop[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF TREND BIAS (4h and 1d HMA) ===
        hma_4h_bull = close[i] > hma_4h_aligned[i]
        hma_4h_bear = close[i] < hma_4h_aligned[i]
        hma_1d_bull = close[i] > hma_1d_aligned[i]
        hma_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === SESSION FILTER (8-20 UTC only) ===
        utc_hour = get_utc_hour(open_time[i])
        in_session = 8 <= utc_hour <= 20
        
        # === CHOPPINESS FILTER (avoid range-bound) ===
        is_trending = chop[i] < 55.0
        
        # === VOLUME FILTER (confirm participation) ===
        volume_confirmed = vol_ratio[i] > 1.2
        
        # === RSI PULLBACK ENTRY LOGIC ===
        # Long: RSI(7) pulled back to 35-45 within uptrend
        # Short: RSI(7) rallied to 55-65 within downtrend
        rsi_long_pullback = 35.0 <= rsi_7[i] <= 48.0
        rsi_short_pullback = 52.0 <= rsi_7[i] <= 65.0
        
        # === DESIRED SIGNAL BASED ON CONFLUENCE ===
        desired_signal = 0.0
        
        # LONG SETUP: All filters must align
        if (hma_4h_bull and hma_1d_bull and  # HTF trend bullish
            is_trending and  # Not choppy
            in_session and  # High liquidity hours
            volume_confirmed and  # Volume confirmation
            rsi_long_pullback):  # RSI pullback entry
            desired_signal = BASE_SIZE
        
        # SHORT SETUP: All filters must align
        elif (hma_4h_bear and hma_1d_bear and  # HTF trend bearish
              is_trending and  # Not choppy
              in_session and  # High liquidity hours
              volume_confirmed and  # Volume confirmation
              rsi_short_pullback):  # RSI pullback entry
            desired_signal = -BASE_SIZE
        
        # === REDUCED SIZE: Missing one HTF confirmation ===
        # Long with only 4h bullish (1d neutral/bear)
        if desired_signal == 0.0:
            if (hma_4h_bull and not hma_1d_bear and
                is_trending and in_session and volume_confirmed and rsi_long_pullback):
                desired_signal = REDUCED_SIZE
            # Short with only 4h bearish (1d neutral/bull)
            elif (hma_4h_bear and not hma_1d_bull and
                  is_trending and in_session and volume_confirmed and rsi_short_pullback):
                desired_signal = -REDUCED_SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
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
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= BASE_SIZE * 0.85:
            final_signal = BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.85:
            final_signal = -BASE_SIZE
        elif desired_signal >= REDUCED_SIZE * 0.85:
            final_signal = REDUCED_SIZE
        elif desired_signal <= -REDUCED_SIZE * 0.85:
            final_signal = -REDUCED_SIZE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals