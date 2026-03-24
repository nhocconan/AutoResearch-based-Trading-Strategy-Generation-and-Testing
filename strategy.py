#!/usr/bin/env python3
"""
Experiment #1508: 30m Primary + 4h/1d HTF — Adaptive Regime HMA+RSI

Hypothesis: After analyzing 1100+ failed strategies, the pattern for lower TF (30m) is clear:
1. Complex filters (CHOP+CRSI+session+volume ALL required) = 0 trades (#1498, #1500)
2. HTF trend bias (1d/4h HMA) + lower TF entry (30m RSI) = positive Sharpe (#1505, #1506)
3. For 30m: MUST limit trades to 30-80/year or fee drag kills profit
4. ADAPTIVE regime: CHOP determines entry TYPE (mean-revert vs trend-follow), not whether to trade
5. Session filter (8-20 UTC) + volume filter reduces noise but doesn't block all entries

Key design choices:
- 1d HMA(21): Macro trend bias (direction filter)
- 4h HMA(21): Intermediate trend confirmation (confluence)
- 30m RSI(14): Pullback entry timing within HTF trend
- 30m CHOP(14): Regime detection - adapts RSI thresholds, doesn't block trades
- Session 8-20 UTC: Avoid Asian session noise
- Volume > 0.8x avg: Confirms momentum
- Position size 0.20 (smaller for 30m trade frequency)
- ATR(14) 2.5x trailing stop for risk management
- LOOSE entry bands to ensure trades happen (RSI 30-70, not 40-60)

Timeframe: 30m (as required by experiment)
HTF: 4h + 1d (dual HTF for stronger trend filter)
Position Size: 0.20 (discrete: 0.0, ±0.20)
Target: 120-320 trades/train (4 years), 30-80 trades/test (15 months), Sharpe > 0.618
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_hma_rsi_chop_regime_4h1d_session_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness
    HMA = WMA(2*WMA(n/2) - WMA(n))
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(data, w_period):
        result = np.full(len(data), np.nan)
        weights = np.arange(1, w_period + 1, dtype=float)
        weights /= weights.sum()
        for i in range(w_period - 1, len(data)):
            if np.any(np.isnan(data[i - w_period + 1:i + 1])):
                continue
            result[i] = np.sum(data[i - w_period + 1:i + 1] * weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan)
    mask = ~np.isnan(wma_half) & ~np.isnan(wma_full)
    diff[mask] = 2.0 * wma_half[mask] - wma_full[mask]
    
    hma = wma(diff, sqrt_period)
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    mask = loss_smooth > 1e-10
    rsi[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi[loss_smooth <= 1e-10] = 100.0
    rsi[:period] = np.nan
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    atr = calculate_atr(high, low, close, period)
    
    chop = np.full(n, np.nan)
    for i in range(period, n):
        atr_sum = np.nansum(atr[i-period+1:i+1])
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_sma(values, period=20):
    """Simple Moving Average"""
    n = len(values)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(values).rolling(window=period, min_periods=period).mean().values
    return sma

def get_hour_from_open_time(open_time):
    """Extract hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds since epoch
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
    
    # Calculate and align HTF HMAs for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate primary (30m) indicators
    hma_30m = calculate_hma(close, period=21)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    vol_sma = calculate_sma(volume, period=20)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.20  # Smaller size for 30m (higher trade frequency potential)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]) or np.isnan(hma_30m[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_4h_aligned[i]) or np.isnan(vol_sma[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (8-20 UTC) - avoid Asian session noise ===
        hour = get_hour_from_open_time(open_time[i])
        in_session = 8 <= hour <= 20
        
        # === VOLUME FILTER ===
        vol_ratio = volume[i] / vol_sma[i] if vol_sma[i] > 1e-10 else 0.0
        vol_ok = vol_ratio >= 0.8
        
        # === MACRO TREND (1d HMA) - primary direction bias ===
        daily_bull = close[i] > hma_1d_aligned[i]
        daily_bear = close[i] < hma_1d_aligned[i]
        
        # === INTERMEDIATE TREND (4h HMA) - confirmation ===
        h4_bull = close[i] > hma_4h_aligned[i]
        h4_bear = close[i] < hma_4h_aligned[i]
        
        # === PRIMARY TREND (30m HMA) - entry confirmation ===
        h30_bull = close[i] > hma_30m[i]
        h30_bear = close[i] < hma_30m[i]
        
        # === CHOPPINESS REGIME - ADAPTS ENTRY THRESHOLDS ===
        # CHOP > 55 = ranging (use mean-reversion thresholds)
        # CHOP < 45 = trending (use trend-follow thresholds)
        # CHOP 45-55 = neutral (use standard thresholds)
        is_ranging = chop[i] > 55.0
        is_trending = chop[i] < 45.0
        
        # === RSI ENTRY THRESHOLDS (ADAPTIVE BY REGIME) ===
        if is_ranging:
            # Mean reversion: buy low, sell high
            rsi_long = rsi[i] < 40.0  # Oversold in range
            rsi_short = rsi[i] > 60.0  # Overbought in range
        elif is_trending:
            # Trend follow: buy pullbacks, sell rallies
            rsi_long = 35.0 <= rsi[i] <= 50.0  # Pullback in uptrend
            rsi_short = 50.0 <= rsi[i] <= 65.0  # Rally in downtrend
        else:
            # Neutral: standard thresholds
            rsi_long = rsi[i] < 45.0
            rsi_short = rsi[i] > 55.0
        
        # === DESIRED SIGNAL - ADAPTIVE BY REGIME + HTF FILTER ===
        desired_signal = 0.0
        
        # LONG entries
        if daily_bull:
            # Strong: 1d + 4h both bull + RSI long
            if h4_bull and rsi_long:
                if in_session and vol_ok:
                    desired_signal = BASE_SIZE
                elif not in_session:
                    desired_signal = BASE_SIZE * 0.5  # Reduced size outside session
            # Medium: 1d bull + 4h bull + 30m HMA bull (trend confirm)
            elif h4_bull and h30_bull and rsi[i] < 55.0:
                if in_session and vol_ok:
                    desired_signal = BASE_SIZE * 0.8
            # Weak: 1d bull only + RSI very low (deep pullback)
            elif rsi[i] < 35.0:
                if in_session:
                    desired_signal = BASE_SIZE * 0.6
        
        # SHORT entries
        elif daily_bear:
            # Strong: 1d + 4h both bear + RSI short
            if h4_bear and rsi_short:
                if in_session and vol_ok:
                    desired_signal = -BASE_SIZE
                elif not in_session:
                    desired_signal = -BASE_SIZE * 0.5
            # Medium: 1d bear + 4h bear + 30m HMA bear (trend confirm)
            elif h4_bear and h30_bear and rsi[i] > 45.0:
                if in_session and vol_ok:
                    desired_signal = -BASE_SIZE * 0.8
            # Weak: 1d bear only + RSI very high (strong rally)
            elif rsi[i] > 65.0:
                if in_session:
                    desired_signal = -BASE_SIZE * 0.6
        
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
        if desired_signal >= BASE_SIZE * 0.9:
            final_signal = BASE_SIZE
        elif desired_signal >= BASE_SIZE * 0.6:
            final_signal = BASE_SIZE * 0.8
        elif desired_signal >= BASE_SIZE * 0.3:
            final_signal = BASE_SIZE * 0.5
        elif desired_signal <= -BASE_SIZE * 0.9:
            final_signal = -BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.6:
            final_signal = -BASE_SIZE * 0.8
        elif desired_signal <= -BASE_SIZE * 0.3:
            final_signal = -BASE_SIZE * 0.5
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
                # Flip position
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