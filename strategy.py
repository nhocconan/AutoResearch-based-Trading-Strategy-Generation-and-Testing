#!/usr/bin/env python3
"""
Experiment #860: 1h Primary + 4h/12h HTF — Regime-Adaptive RSI Pullback

Hypothesis: 1h strategies failed (#850, #855) due to EITHER 0 trades (too strict)
OR too many trades (fee drag). Solution: Use 4h/12h for DIRECTION, 1h for TIMING.

Key design:
1. 4h HMA(21) = trend bias (only long if price>4h_HMA, only short if price<4h_HMA)
2. 12h HMA(21) = secular filter (adds conviction when aligned with 4h)
3. 1h RSI(14) = entry trigger (pullback to 35-45 for long, 55-65 for short)
4. 1h Choppiness(14) = regime filter (CHOP>50 mean revert, CHOP<50 trend follow)
5. 1h Volume > 0.8x 20-avg = confirmation (avoids low-liquidity traps)
6. Session filter 8-20 UTC = reduces trades by ~60% (critical for 1h)
7. ATR(14) trailing stop 2.5x = risk management

Why this should work:
- 4h/12h alignment = fewer false signals (HTF confluence)
- RSI pullback (not extreme) = more trades than RSI<30/>70
- Session filter = avoids Asian session chop
- Hold logic = maintains position through minor pullbacks
- Fallback entries = guarantees trades on all symbols

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 1h (target 40-70 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_rsi_pullback_4h12h_hma_chop_vol_session_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_sma(series, period):
    """Simple Moving Average."""
    return pd.Series(series).rolling(window=period, min_periods=period).mean().values

def calculate_hma(series, period):
    """Hull Moving Average."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppy vs trending.
    CHOP > 55 = ranging, CHOP < 45 = trending.
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100
            continue
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], np.abs(high[j] - close[j-1]), np.abs(low[j] - close[j-1]))
            tr_sum += tr
        
        chop[i] = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
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
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[j] - close[j-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_avg(volume, period=20):
    """Average volume over period."""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def get_hour_from_open_time(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
    return (open_time // (1000 * 60 * 60)) % 24

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
    
    # Calculate primary (1h) indicators
    rsi_1h = calculate_rsi(close, period=14)
    chop_1h = calculate_choppiness(high, low, close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    vol_avg_1h = calculate_volume_avg(volume, period=20)
    sma_200_1h = calculate_sma(close, 200)
    
    # Calculate and align 4h HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 12h HMA for secular filter
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.15
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(rsi_1h[i]) or np.isnan(chop_1h[i]) or np.isnan(atr_1h[i]):
            continue
        if np.isnan(vol_avg_1h[i]) or vol_avg_1h[i] <= 1e-10:
            continue
        if atr_1h[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            continue
        if np.isnan(sma_200_1h[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        hour = get_hour_from_open_time(open_time[i])
        in_session = 8 <= hour <= 20
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] >= 0.8 * vol_avg_1h[i]
        
        # === HTF TREND BIAS (4h HMA21) ===
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # === SECULAR FILTER (12h HMA21) ===
        trend_12h_bullish = close[i] > hma_12h_aligned[i]
        trend_12h_bearish = close[i] < hma_12h_aligned[i]
        
        # === HTF CONFLUENCE ===
        strong_bullish = trend_4h_bullish and trend_12h_bullish
        strong_bearish = trend_4h_bearish and trend_12h_bearish
        neutral_htf = (trend_4h_bullish and trend_12h_bearish) or (trend_4h_bearish and trend_12h_bullish)
        
        # === REGIME DETECTION (1h Choppiness Index) ===
        ranging_regime = chop_1h[i] > 50
        trending_regime = chop_1h[i] < 50
        
        # === RSI SIGNALS (Pullback levels, not extremes) ===
        rsi_oversold_pullback = 35 <= rsi_1h[i] <= 45
        rsi_overbought_pullback = 55 <= rsi_1h[i] <= 65
        rsi_extreme_oversold = rsi_1h[i] < 30
        rsi_extreme_overbought = rsi_1h[i] > 70
        rsi_neutral_bull = 45 < rsi_1h[i] < 55
        rsi_rising = rsi_1h[i] > rsi_1h[i-1] if i > 0 and not np.isnan(rsi_1h[i-1]) else False
        rsi_falling = rsi_1h[i] < rsi_1h[i-1] if i > 0 and not np.isnan(rsi_1h[i-1]) else False
        
        desired_signal = 0.0
        
        # === RANGING REGIME (CHOP > 50) — Mean Reversion ===
        if ranging_regime and in_session and volume_ok:
            # Long: Strong bullish HTF + RSI pullback + RSI rising
            if strong_bullish and rsi_oversold_pullback and rsi_rising:
                desired_signal = BASE_SIZE
            # Short: Strong bearish HTF + RSI pullback + RSI falling
            elif strong_bearish and rsi_overbought_pullback and rsi_falling:
                desired_signal = -BASE_SIZE
            # Fallback: Extreme RSI + any HTF alignment
            elif rsi_extreme_oversold and trend_4h_bullish:
                desired_signal = REDUCED_SIZE
            elif rsi_extreme_overbought and trend_4h_bearish:
                desired_signal = -REDUCED_SIZE
        
        # === TRENDING REGIME (CHOP < 50) — Trend Following ===
        elif trending_regime and in_session and volume_ok:
            # Long: Strong bullish HTF + RSI neutral/rising + above SMA200
            if strong_bullish and (rsi_neutral_bull or rsi_rising) and close[i] > sma_200_1h[i]:
                desired_signal = BASE_SIZE
            # Short: Strong bearish HTF + RSI neutral/falling + below SMA200
            elif strong_bearish and (rsi_neutral_bull or rsi_falling) and close[i] < sma_200_1h[i]:
                desired_signal = -BASE_SIZE
            # Fallback: HTF trend + RSI pullback
            elif trend_4h_bullish and rsi_oversold_pullback:
                desired_signal = REDUCED_SIZE
            elif trend_4h_bearish and rsi_overbought_pullback:
                desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL HTF (4h/12h disagreement) — Conservative ===
        if neutral_htf and in_session and volume_ok:
            # Only take extreme RSI signals
            if rsi_extreme_oversold and rsi_rising:
                desired_signal = REDUCED_SIZE if desired_signal == 0 else desired_signal
            elif rsi_extreme_overbought and rsi_falling:
                desired_signal = -REDUCED_SIZE if desired_signal == 0 else desired_signal
        
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
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if 4h trend intact and RSI not overbought
                if trend_4h_bullish and rsi_1h[i] < 65:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if 4h trend intact and RSI not oversold
                if trend_4h_bearish and rsi_1h[i] > 35:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if 4h trend reverses
            if trend_4h_bearish and rsi_1h[i] > 65:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 4h trend reverses
            if trend_4h_bullish and rsi_1h[i] < 35:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE else -REDUCED_SIZE
        
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