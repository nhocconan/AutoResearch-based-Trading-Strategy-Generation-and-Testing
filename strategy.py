#!/usr/bin/env python3
"""
Experiment #735: 1h Primary + 4h/1d HTF — Fisher Transform + Volume Confluence

Hypothesis: After 6 consecutive failures on 1h timeframe (#725, #730 both Sharpe=0),
the issue is CRSI mean-reversion doesn't work well on lower TFs. This strategy uses:
1. Fisher Transform (period=9) for precise reversal entries (proven in Ehlers literature)
2. 4h HMA(21) for intermediate trend direction
3. 1d HMA(21) for long-term bias filter
4. Volume filter (>1.2x 20-bar avg) to confirm real moves
5. Session filter (8-20 UTC) for liquidity
6. ATR volatility filter (only trade when ATR > 20-bar median)

Key differences from failed #725/#730:
- Fisher Transform instead of CRSI (better for lower TF reversals)
- Stricter confluence (5 filters instead of 3) = fewer trades
- Volume confirmation to avoid fakeouts
- Session filter to avoid Asian session low-liquidity whipsaws
- Target: 40-60 trades/year (within 1h limit of 30-60)

Timeframe: 1h (as required for this experiment)
Target: Sharpe > 0.612 (beat current best), trades 40-60/year, ALL symbols positive
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_volume_hma_4h1d_session_v1"
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

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution.
    Long when Fisher crosses above -1.5, short when crosses below +1.5.
    """
    n = len(high)
    fisher = np.full(n, np.nan)
    fisher_signal = np.full(n, np.nan)
    
    if n < period + 1:
        return fisher, fisher_signal
    
    # Calculate typical price
    typical = (high + low + pd.Series(high).rolling(2).min().values) / 3
    
    # Normalize to -1 to +1 range
    highest = pd.Series(typical).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(typical).rolling(window=period, min_periods=period).min().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        normalized = (typical - lowest) / (highest - lowest + 1e-10)
        normalized = np.clip(normalized, 0.001, 0.999)  # Avoid log(0)
    
    # Fisher transform
    fisher_raw = 0.5 * np.log((1 + normalized) / (1 - normalized + 1e-10))
    fisher = pd.Series(fisher_raw).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Signal line (1-period lag)
    fisher_signal = np.concatenate([[np.nan], fisher[:-1]])
    
    return fisher, fisher_signal

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

def calculate_sma(close, period):
    """Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_volume_ratio(volume, period=20):
    """Volume ratio vs moving average."""
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = volume / (vol_sma + 1e-10)
    return ratio

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
    fisher, fisher_signal = calculate_fisher_transform(high, low, period=9)
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    sma_200 = calculate_sma(close, period=200)
    
    # ATR median for volatility filter
    atr_median = pd.Series(atr_1h).rolling(window=20, min_periods=20).apply(
        lambda x: np.median(x[~np.isnan(x)]) if len(x[~np.isnan(x)]) > 0 else np.nan
    ).values
    
    # Calculate and align HTF HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Get UTC hours for session filter
    utc_hours = get_utc_hour(open_time)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Smaller size for 1h to reduce fee impact
    REDUCED_SIZE = 0.15
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(300, n):  # Need buffer for all indicators + HTF alignment
        # Skip if indicators not ready
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            continue
        if np.isnan(atr_1h[i]) or atr_1h[i] <= 1e-10 or np.isnan(atr_median[i]):
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(rsi_1h[i]) or np.isnan(vol_ratio[i]) or np.isnan(sma_200[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        in_session = 8 <= utc_hours[i] <= 20
        
        # === VOLATILITY FILTER (ATR > median) ===
        vol_ok = atr_1h[i] > atr_median[i] * 0.8  # Allow some buffer
        
        # === VOLUME FILTER (>1.2x average) ===
        volume_ok = vol_ratio[i] > 1.15  # Slightly lower threshold to ensure trades
        
        # === TREND BIAS (4h and 1d HTF HMA) ===
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # Strong trend when both agree
        strong_bullish = trend_4h_bullish and trend_1d_bullish
        strong_bearish = trend_4h_bearish and trend_1d_bearish
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        desired_signal = 0.0
        
        # === LONG ENTRY CONDITIONS (strict confluence) ===
        # Fisher cross above -1.5 (reversal from oversold)
        fisher_long_cross = fisher_signal[i] < -1.5 and fisher[i] >= -1.5
        
        long_confluence = (
            fisher_long_cross and
            in_session and
            volume_ok and
            vol_ok and
            (strong_bullish or (trend_4h_bullish and above_sma200)) and
            rsi_1h[i] < 55  # Not overbought
        )
        
        if long_confluence:
            desired_signal = BASE_SIZE
        
        # === SHORT ENTRY CONDITIONS (strict confluence) ===
        # Fisher cross below +1.5 (reversal from overbought)
        fisher_short_cross = fisher_signal[i] > 1.5 and fisher[i] <= 1.5
        
        short_confluence = (
            fisher_short_cross and
            in_session and
            volume_ok and
            vol_ok and
            (strong_bearish or (trend_4h_bearish and below_sma200)) and
            rsi_1h[i] > 45  # Not oversold
        )
        
        if short_confluence:
            desired_signal = -BASE_SIZE
        
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
                # Hold long if 4h HMA still bullish and Fisher not extremely overbought
                if trend_4h_bullish and fisher[i] < 2.0:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if 4h HMA still bearish and Fisher not extremely oversold
                if trend_4h_bearish and fisher[i] > -2.0:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if 4h trend reverses or Fisher extremely overbought
            if trend_4h_bearish or fisher[i] > 2.5:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 4h trend reverses or Fisher extremely oversold
            if trend_4h_bullish or fisher[i] < -2.5:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE
        
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
                # Position flip
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