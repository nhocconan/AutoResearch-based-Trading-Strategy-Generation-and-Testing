#!/usr/bin/env python3
"""
Experiment #715: 1h Primary + 4h/1d HTF — Regime-Adaptive Trend Following

Hypothesis: Lower TF (1h) strategies fail due to fee drag from too many trades.
Solution: Use 4h/1d HTF for TREND DIRECTION, 1h only for ENTRY TIMING.
This gives HTF trade frequency (30-60/year) with 1h execution precision.

Key components:
1. 4h HMA(21) = primary trend bias (load ONCE before loop)
2. 1d HMA(21) = meta-trend filter (stronger bias confirmation)
3. 1h Choppiness Index(14) = regime detection (>55 range, <45 trend)
4. 1h RSI(7) = entry timing (pullback in trend, extremes in range)
5. Session filter (8-20 UTC) = avoid low liquidity whipsaws
6. Volume filter (>0.8x 20-bar avg) = confirm participation
7. ATR(14) trailing stop = risk management

Why this should work:
- 1h TF allows precise entry timing within 4h trend
- HTF filters reduce trade frequency to acceptable levels
- Choppiness adapts logic to market regime
- Session/volume filters avoid false signals
- Conservative sizing (0.25) protects against 2022-style crashes

Target: Sharpe > 0.612, trades 30-80/year, ALL symbols positive
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_hma_rsi_chop_session_4h1d_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(series, period):
    """Hull Moving Average - smoother and more responsive than EMA."""
    if len(series) < period:
        return np.full(len(series), np.nan)
    
    wma1 = pd.Series(series).ewm(span=period//2, min_periods=period//2, adjust=False).mean().values
    wma2 = pd.Series(series).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    hma_raw = 2 * wma1 - wma2
    hma = pd.Series(hma_raw).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean().values
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending.
    Formula: 100 * (ATR(1) sum / ATR(period)) / (log(HH - LL) / log(period))
    CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    # ATR sum over period
    atr_sum = np.zeros(n)
    for i in range(period - 1, n):
        atr_sum[i] = np.sum(tr[i-period+1:i+1])
    
    # ATR over period
    atr_period = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Highest High and Lowest Low over period
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Choppiness calculation
    with np.errstate(divide='ignore', invalid='ignore'):
        chop_raw = 100 * (atr_sum / (atr_period * period + 1e-10)) / (np.log(hh - ll + 1e-10) / np.log(period) + 1e-10)
        chop = np.clip(chop_raw, 0, 100)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_sma(close, period):
    """Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
    return pd.to_datetime(open_time, unit='ms').dt.hour.values

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
    
    # Calculate primary (1h) indicators
    rsi_1h = calculate_rsi(close, period=7)  # Faster RSI for entry timing
    chop_1h = calculate_choppiness_index(high, low, close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    sma_vol_20 = calculate_sma(volume, period=20)
    utc_hour = get_utc_hour(open_time)
    
    # Calculate and align HTF HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Conservative for 1h TF
    REDUCED_SIZE = 0.15
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):  # Need buffer for HTF alignment + indicators
        # Skip if indicators not ready
        if np.isnan(rsi_1h[i]) or np.isnan(atr_1h[i]) or np.isnan(chop_1h[i]):
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(sma_vol_20[i]) or atr_1h[i] <= 1e-10:
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        in_session = 8 <= utc_hour[i] <= 20
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * sma_vol_20[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        regime_trend = chop_1h[i] < 45  # Trending market
        regime_range = chop_1h[i] > 55  # Ranging market
        # 45-55 = neutral, use trend logic with stricter filters
        
        # === TREND BIAS (4h + 1d HMA) ===
        # Strong bullish: price > 4h HMA > 1d HMA
        # Strong bearish: price < 4h HMA < 1d HMA
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # Confluence: both 4h and 1d agree
        trend_strong_bullish = trend_4h_bullish and trend_1d_bullish
        trend_strong_bearish = trend_4h_bearish and trend_1d_bearish
        
        desired_signal = 0.0
        
        # === TREND REGIME (CHOP < 45) ===
        if regime_trend:
            # Long: Strong bullish trend + RSI pullback (30-45)
            if trend_strong_bullish and 30 <= rsi_1h[i] <= 45 and in_session and volume_ok:
                desired_signal = BASE_SIZE
            
            # Short: Strong bearish trend + RSI bounce (55-70)
            elif trend_strong_bearish and 55 <= rsi_1h[i] <= 70 and in_session and volume_ok:
                desired_signal = -BASE_SIZE
            
            # Weaker signal (4h only, reduced size)
            elif trend_4h_bullish and rsi_1h[i] < 35 and in_session:
                desired_signal = REDUCED_SIZE
            elif trend_4h_bearish and rsi_1h[i] > 65 and in_session:
                desired_signal = -REDUCED_SIZE
        
        # === RANGE REGIME (CHOP > 55) ===
        elif regime_range:
            # Long: RSI deeply oversold (<25) + session + volume
            if rsi_1h[i] < 25 and in_session and volume_ok:
                desired_signal = REDUCED_SIZE
            
            # Short: RSI deeply overbought (>75) + session + volume
            elif rsi_1h[i] > 75 and in_session and volume_ok:
                desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55) ===
        else:
            # Only take strongest trend signals with full confluence
            if trend_strong_bullish and rsi_1h[i] < 30 and in_session and volume_ok:
                desired_signal = REDUCED_SIZE
            elif trend_strong_bearish and rsi_1h[i] > 70 and in_session and volume_ok:
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
        
        # === HOLD LOGIC — Maintain position if trend intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if 4h trend still bullish and RSI not overbought
                if trend_4h_bullish and rsi_1h[i] < 70:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if 4h trend still bearish and RSI not oversold
                if trend_4h_bearish and rsi_1h[i] > 30:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if trend reverses or RSI overbought
            if rsi_1h[i] > 75 or close[i] < hma_4h_aligned[i]:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if trend reverses or RSI oversold
            if rsi_1h[i] < 25 or close[i] > hma_4h_aligned[i]:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE * 0.8 else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE * 0.8 else -REDUCED_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1h[i]
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
        
        signals[i] = desired_signal
    
    return signals