#!/usr/bin/env python3
"""
Experiment #480: 1h Primary + 4h/12h HTF — HMA Trend + RSI Pullback + Choppiness Regime

Hypothesis: Based on proven pattern from best strategy (mtf_4h_triple_regime_crsi_donchian_1d1w_v1),
using HTF for trend direction and LTF for entry timing. Key innovations for 1h timeframe:
1. 4h HMA(21) for major trend bias (proven stable trend filter)
2. 12h HMA(21) for higher-order trend confirmation (avoid counter-trend trades)
3. 1h RSI(14) for pullback entries (simpler than CRSI, more reliable trade generation)
4. Choppiness Index(14) regime filter: CHOP>55=range (mean revert), CHOP<45=trend (follow)
5. Volume filter: volume > 0.8x 20-bar average (avoid low-liquidity entries)
6. Session filter: only trade 8-20 UTC (high liquidity hours)
7. Relaxed RSI thresholds: <40/>60 (not extreme 25/75) to ensure trade generation
8. Hold logic: maintain position while HTF trend intact (reduce churn)
9. ATR(14) trailing stop at 2.5x for risk management
10. Discrete sizing: 0.0, ±0.25, ±0.30 to minimize fee churn

Why this should work for 1h:
- 4h/12h HTF ensures we only trade with major trend (reduces whipsaws)
- 1h RSI pullbacks give precise entry timing within HTF trend
- Choppiness regime adapts logic: trend-follow in trends, mean-revert in ranges
- Relaxed RSI thresholds ensure we generate 30-80 trades/year (not 0 trades)
- Session + volume filters avoid low-quality entries during illiquid hours
- Hold logic prevents premature exits during pullbacks

Target: Sharpe > 0.612, DD < -35%, trades >= 30 on train, >= 3 on test, ALL symbols positive Sharpe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_chop_regime_4h12h_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    Reduces lag while maintaining smoothness.
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    # WMA helper
    def wma(series, span):
        weights = np.arange(1, span + 1)
        weights = weights / weights.sum()
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            result[i] = np.sum(series[i - span + 1:i + 1] * weights)
        return result
    
    close_s = pd.Series(close)
    wma_half = wma(close, period // 2)
    wma_full = wma(close, period)
    
    diff = 2.0 * wma_half - wma_full
    hma = wma(diff, int(np.sqrt(period)))
    
    return hma

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = range/chop, CHOP < 38.2 = trending
    Using 55/45 thresholds for clearer regime separation.
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        
        if highest - lowest < 1e-10:
            chop[i] = 50.0
            continue
        
        tr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr1 = high[j] - low[j]
            tr2 = np.abs(high[j] - close[j - 1]) if j > 0 else tr1
            tr3 = np.abs(low[j] - close[j - 1]) if j > 0 else tr1
            tr_sum += max(tr1, tr2, tr3)
        
        if tr_sum > 1e-10:
            chop[i] = 100.0 * np.log10((highest - lowest) / tr_sum) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing method."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0.0)
    loss[1:] = np.where(delta < 0, -delta, 0.0)
    
    gain_s = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_s = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = gain_s / (loss_s + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_avg(volume, period=20):
    """Calculate rolling average volume."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_avg

def get_hour_from_open_time(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
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
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 1h indicators (primary timeframe)
    rsi_1h = calculate_rsi(close, period=14)
    chop_1h = calculate_choppiness(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    vol_avg_20 = calculate_volume_avg(volume, period=20)
    
    # Calculate and align HTF indicators (4h HMA for trend)
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 12h HMA (higher-order trend)
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.30
    SIZE_SHORT = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(rsi_1h[i]):
            continue
        if np.isnan(chop_1h[i]):
            continue
        if np.isnan(vol_avg_20[i]) or vol_avg_20[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]):
            continue
        if np.isnan(hma_12h_aligned[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        hour = get_hour_from_open_time(open_time[i])
        in_session = 8 <= hour <= 20
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * vol_avg_20[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_chop = chop_1h[i] > 55.0  # Range/mean reversion regime
        is_trend = chop_1h[i] < 45.0  # Trending regime
        # Neutral zone: 45 <= CHOP <= 55
        
        # === HTF TREND BIAS (4h HMA) ===
        htf_4h_bullish = close[i] > hma_4h_aligned[i]
        htf_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # === HIGHER-ORDER TREND (12h HMA) ===
        htf_12h_bullish = close[i] > hma_12h_aligned[i]
        htf_12h_bearish = close[i] < hma_12h_aligned[i]
        
        # === RSI SIGNALS (relaxed thresholds for trade generation) ===
        rsi_oversold = rsi_1h[i] < 40.0
        rsi_overbought = rsi_1h[i] > 60.0
        rsi_extreme_oversold = rsi_1h[i] < 30.0
        rsi_extreme_overbought = rsi_1h[i] > 70.0
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRIES
        long_score = 0
        
        # HTF bias alignment (4h required, 12h bonus)
        if htf_4h_bullish:
            long_score += 2
        if htf_12h_bullish:
            long_score += 1
        
        # RSI entry signal (different logic per regime)
        if is_trend:
            # In trend: RSI pullback to oversold
            if rsi_oversold:
                long_score += 2
        elif is_chop:
            # In chop: RSI extreme oversold for mean reversion
            if rsi_extreme_oversold:
                long_score += 2
        else:
            # Neutral: moderate RSI oversold
            if rsi_oversold:
                long_score += 1
        
        # Session and volume filters (required)
        if in_session and volume_ok:
            if long_score >= 4:
                desired_signal = SIZE_LONG
        
        # SHORT ENTRIES
        if desired_signal == 0.0:
            short_score = 0
            
            # HTF bias alignment
            if htf_4h_bearish:
                short_score += 2
            if htf_12h_bearish:
                short_score += 1
            
            # RSI entry signal
            if is_trend:
                if rsi_overbought:
                    short_score += 2
            elif is_chop:
                if rsi_extreme_overbought:
                    short_score += 2
            else:
                if rsi_overbought:
                    short_score += 1
            
            # Session and volume filters
            if in_session and volume_ok:
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
        
        # === HOLD LOGIC — Maintain position if HTF trend unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and htf_4h_bullish:
                desired_signal = SIZE_LONG
            elif position_side < 0 and htf_4h_bearish:
                desired_signal = -SIZE_SHORT
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = 0.30
        elif desired_signal < 0:
            desired_signal = -0.25
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
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