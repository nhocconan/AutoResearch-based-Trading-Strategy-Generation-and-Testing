#!/usr/bin/env python3
"""
Experiment #705: 1h Primary + 4h/1d HTF — Choppiness Regime + RSI Pullback + Volume

Hypothesis: Lower timeframe (1h) can work IF we use HTF for direction and only trade
during high-probability windows. Key insight from failures: entry conditions were TOO
STRICT (0 trades on #695, #700). This strategy uses RELAXED but meaningful filters:

1. 4h HMA(21) for trend direction (not both 4h+1d, just 4h for more signals)
2. 1d Choppiness for regime: CHOP>55=range(mean-revert), CHOP<45=trend(follow)
3. 1h RSI(14) for entry: 25/75 thresholds (not 15/85 - too strict)
4. Volume filter: >0.7x 20-bar avg (not 1.5x - too strict)
5. Session filter: 8-20 UTC ONLY (reduces trades but increases quality)
6. ATR stoploss: 2.5x trailing

Why this should work:
- 1h TF worked in #682 era before we over-complicated
- Relaxed thresholds ensure 30-80 trades/year (not 0 like #700)
- Session filter reduces low-quality Asia session trades
- 4h HMA alone (not 4h+1d) = more signals while keeping trend bias
- Discrete sizing (0.0, ±0.25) minimizes fee churn

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_chop_regime_rsi_4h_session_v1"
timeframe = "1h"
leverage = 1.0

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppy vs trending.
    CHOP > 61.8 = range-bound (mean reversion)
    CHOP < 38.2 = trending (trend following)
    Source: E.W. Dreiss
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        tr_sum = 0.0
        highest_high = high[i-period+1]
        lowest_low = low[i-period+1]
        
        for j in range(i-period+1, i+1):
            tr_sum += max(high[j] - low[j], 
                         abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j],
                         abs(low[j] - close[j-1]) if j > 0 else high[j] - low[j])
            highest_high = max(highest_high, high[j])
            lowest_low = min(lowest_low, low[j])
        
        atr_like = tr_sum / period
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_like > 1e-10:
            chop[i] = 100 * np.log10(price_range / (atr_like * np.sqrt(period))) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return np.clip(chop, 0, 100)

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
    
    return np.clip(rsi, 0, 100)

def calculate_hma(series, period):
    """Hull Moving Average."""
    if len(series) < period:
        return np.full(len(series), np.nan)
    
    wma1 = pd.Series(series).ewm(span=period//2, min_periods=period//2, adjust=False).mean().values
    wma2 = pd.Series(series).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    hma_raw = 2 * wma1 - wma2
    hma = pd.Series(hma_raw).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean().values
    return hma

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], 
                   abs(high[i] - close[i-1]), 
                   abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_sma(volume, period=20):
    """Volume simple moving average."""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def get_hour_from_timestamp(open_time):
    """Extract UTC hour from open_time (milliseconds)."""
    return (open_time // 3600000) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (CRITICAL - Rule 1)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (1h) indicators
    rsi_1h = calculate_rsi(close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    vol_sma_1h = calculate_volume_sma(volume, period=20)
    
    # Calculate HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    chop_1d_raw = calculate_choppiness(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Smaller for 1h TF (fee sensitivity)
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(rsi_1h[i]) or np.isnan(atr_1h[i]) or np.isnan(vol_sma_1h[i]):
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(chop_1d_aligned[i]):
            continue
        if atr_1h[i] <= 1e-10:
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        hour = get_hour_from_timestamp(open_time[i])
        in_session = 8 <= hour <= 20
        
        # === VOLUME FILTER (relaxed: >0.7x avg) ===
        volume_ok = volume[i] > 0.7 * vol_sma_1h[i]
        
        # === REGIME (Choppiness from 1d) ===
        chop = chop_1d_aligned[i]
        is_range = chop > 55  # Mean reversion regime
        is_trend = chop < 45  # Trend following regime
        is_neutral = not is_range and not is_trend
        
        # === TREND BIAS (4h HMA) ===
        trend_bullish = close[i] > hma_4h_aligned[i]
        trend_bearish = close[i] < hma_4h_aligned[i]
        
        # === RSI SIGNALS (relaxed thresholds) ===
        rsi = rsi_1h[i]
        rsi_oversold = rsi < 30
        rsi_overbought = rsi > 70
        rsi_extreme_oversold = rsi < 25
        rsi_extreme_overbought = rsi > 75
        
        desired_signal = 0.0
        
        # Only trade during session with volume confirmation
        if in_session and volume_ok:
            # === RANGE REGIME (Mean Reversion) ===
            if is_range:
                # Long: RSI oversold + bullish or neutral 4h trend
                if rsi_oversold and (trend_bullish or is_neutral):
                    desired_signal = BASE_SIZE
                # Short: RSI overbought + bearish or neutral 4h trend
                elif rsi_overbought and (trend_bearish or is_neutral):
                    desired_signal = -BASE_SIZE
                # Extreme RSI overrides trend (strong mean reversion)
                elif rsi_extreme_oversold:
                    desired_signal = BASE_SIZE * 0.5
                elif rsi_extreme_overbought:
                    desired_signal = -BASE_SIZE * 0.5
            
            # === TREND REGIME (Trend Following) ===
            elif is_trend:
                # Long pullback: bullish trend + RSI dipping but not oversold
                if trend_bullish and 35 < rsi < 60:
                    desired_signal = BASE_SIZE
                # Short pullback: bearish trend + RSI rising but not overbought
                elif trend_bearish and 40 < rsi < 65:
                    desired_signal = -BASE_SIZE
            
            # === NEUTRAL REGIME (Light signals) ===
            elif is_neutral:
                # Only take extreme RSI with trend confirmation
                if rsi_extreme_oversold and trend_bullish:
                    desired_signal = BASE_SIZE * 0.5
                elif rsi_extreme_overbought and trend_bearish:
                    desired_signal = -BASE_SIZE * 0.5
        
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
        
        # === HOLD LOGIC ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if RSI not extremely overbought
                if rsi < 80:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if RSI not extremely oversold
                if rsi > 20:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            if rsi > 80:  # Overbought exit
                desired_signal = 0.0
            elif close[i] < hma_4h_aligned[i]:  # Trend reversal
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            if rsi < 20:  # Oversold exit
                desired_signal = 0.0
            elif close[i] > hma_4h_aligned[i]:  # Trend reversal
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            if desired_signal >= BASE_SIZE * 0.8:
                desired_signal = BASE_SIZE
            else:
                desired_signal = BASE_SIZE * 0.5
        elif desired_signal < 0:
            if desired_signal <= -BASE_SIZE * 0.8:
                desired_signal = -BASE_SIZE
            else:
                desired_signal = -BASE_SIZE * 0.5
        
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