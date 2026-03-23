#!/usr/bin/env python3
"""
Experiment #780: 1h Primary + 4h/12h HTF — Simplified Trend-Pullback with Session Filter

Hypothesis: After analyzing failures #770 (1h RSI+BB Sharpe=-1.108) and #775 (1h CRSI Sharpe=0.248):
1. Lower TF strategies fail from too many trades → fee drag destroys profit
2. Connors RSI is too complex and generates marginal signals on 1h
3. Need SIMPLER entry: RSI(7) extremes + HTF trend + session filter
4. 12h HMA(21) provides cleaner trend signal than EMA50 (less lag, smoother)
5. Session filter (8-20 UTC) captures high-liquidity periods, reduces noise
6. Volume filter should be RELAXED (0.8x) — we filter via session + HTF trend instead
7. Discrete signals (0.0, ±0.25, ±0.30) minimize churn costs

Strategy design:
1. 12h HMA(21) for trend bias (aligned via mtf_data helper)
2. 1h RSI(7) for entry timing (faster than RSI14, catches pullbacks)
3. 1h Bollinger Bands (20, 2.0) for mean reversion confirmation
4. Session filter: only trade 8-20 UTC (high liquidity, less noise)
5. Volume filter: relaxed to 0.8x avg (don't over-filter)
6. ATR(14) trailing stop: 2.5x for risk management
7. Position sizing: 0.25 base, 0.30 with full confluence

Key improvements from failed 1h strategies:
- Simpler RSI(7) vs Connors RSI (fewer false signals)
- Session filter replaces complex regime detection
- Relaxed volume filter (0.8x vs 1.2x) for adequate trade frequency
- HMA(21) vs EMA(50) for faster trend response on 12h
- Discrete sizing reduces churn

Target: Sharpe > 0.612, trades 30-80/year, ALL symbols positive
Timeframe: 1h (target 30-60 trades/year per Rule 10)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_rsi_hma_session_bb_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(series, period):
    """Hull Moving Average - smoother, less lag than EMA."""
    series = pd.Series(series)
    wma1 = series.rolling(window=period//2, min_periods=period//2).mean()
    wma2 = series.rolling(window=period, min_periods=period).mean()
    wma_diff = 2 * wma1 - wma2
    hma = wma_diff.rolling(window=int(np.sqrt(period)), min_periods=int(np.sqrt(period))).mean()
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

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, lower, sma

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

def calculate_volume_sma(volume, period=20):
    """Simple Moving Average of volume."""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
    dt = pd.to_datetime(open_time, unit='ms', utc=True)
    return dt.hour

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
    rsi_1h = calculate_rsi(close, period=7)
    atr_1h = calculate_atr(high, low, close, period=14)
    bb_upper, bb_lower, bb_sma = calculate_bollinger(close, period=20, std_mult=2.0)
    vol_sma_1h = calculate_volume_sma(volume, period=20)
    
    # Calculate and align HTF HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    FULL_SIZE = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(rsi_1h[i]) or np.isnan(atr_1h[i]) or atr_1h[i] <= 1e-10:
            continue
        if np.isnan(bb_sma[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        if np.isnan(vol_sma_1h[i]) or vol_sma_1h[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        utc_hour = get_utc_hour(open_time[i])
        in_session = 8 <= utc_hour <= 20
        
        # === TREND BIAS (12h + 4h HMA21) ===
        trend_12h_bullish = close[i] > hma_12h_aligned[i]
        trend_12h_bearish = close[i] < hma_12h_aligned[i]
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # Strong trend: both 12h and 4h agree
        strong_bullish = trend_12h_bullish and trend_4h_bullish
        strong_bearish = trend_12h_bearish and trend_4h_bearish
        
        # === VOLUME CONFIRMATION (relaxed) ===
        volume_confirmed = volume[i] > 0.8 * vol_sma_1h[i]
        
        # === RSI SIGNALS (7-period, faster) ===
        rsi_oversold = rsi_1h[i] < 30
        rsi_overbought = rsi_1h[i] > 70
        rsi_extreme_oversold = rsi_1h[i] < 20
        rsi_extreme_overbought = rsi_1h[i] > 80
        rsi_neutral_long = 35 < rsi_1h[i] < 50
        rsi_neutral_short = 50 < rsi_1h[i] < 65
        
        # === BOLLINGER POSITION ===
        below_bb_lower = close[i] <= bb_lower[i]
        above_bb_upper = close[i] >= bb_upper[i]
        near_bb_lower = bb_lower[i] < close[i] <= bb_lower[i] * 1.01
        near_bb_upper = bb_upper[i] > close[i] >= bb_upper[i] * 0.99
        
        desired_signal = 0.0
        confluence_count = 0
        
        # === LONG ENTRY CONDITIONS ===
        long_conditions = []
        
        # Condition 1: RSI oversold
        if rsi_oversold:
            long_conditions.append(True)
        else:
            long_conditions.append(False)
        
        # Condition 2: Price at/near BB lower (mean reversion)
        if below_bb_lower or near_bb_lower:
            long_conditions.append(True)
        else:
            long_conditions.append(False)
        
        # Condition 3: HTF trend not bearish (12h HMA)
        if not trend_12h_bearish:
            long_conditions.append(True)
        else:
            long_conditions.append(False)
        
        # Condition 4: In session (8-20 UTC)
        if in_session:
            long_conditions.append(True)
        else:
            long_conditions.append(False)
        
        confluence_count = sum(long_conditions)
        
        # Enter long with 3+ confluence
        if confluence_count >= 3:
            if volume_confirmed:
                desired_signal = FULL_SIZE
            else:
                desired_signal = BASE_SIZE
        
        # Strong bullish trend + RSI pullback (trend continuation)
        if strong_bullish and rsi_neutral_long and not in_position:
            if volume_confirmed and in_session:
                desired_signal = max(desired_signal, BASE_SIZE)
        
        # === SHORT ENTRY CONDITIONS ===
        short_conditions = []
        
        # Condition 1: RSI overbought
        if rsi_overbought:
            short_conditions.append(True)
        else:
            short_conditions.append(False)
        
        # Condition 2: Price at/near BB upper (mean reversion)
        if above_bb_upper or near_bb_upper:
            short_conditions.append(True)
        else:
            short_conditions.append(False)
        
        # Condition 3: HTF trend not bullish (12h HMA)
        if not trend_12h_bullish:
            short_conditions.append(True)
        else:
            short_conditions.append(False)
        
        # Condition 4: In session (8-20 UTC)
        if in_session:
            short_conditions.append(True)
        else:
            short_conditions.append(False)
        
        confluence_count_short = sum(short_conditions)
        
        # Enter short with 3+ confluence
        if confluence_count_short >= 3:
            if volume_confirmed:
                desired_signal = -FULL_SIZE if desired_signal == 0 else desired_signal
            else:
                desired_signal = -BASE_SIZE if desired_signal == 0 else desired_signal
        
        # Strong bearish trend + RSI pullback (trend continuation)
        if strong_bearish and rsi_neutral_short and not in_position:
            if volume_confirmed and in_session:
                if desired_signal == 0:
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
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if trend reverses strongly
            if strong_bearish:
                desired_signal = 0.0
            # Exit if RSI overbought (take profit)
            if rsi_overbought:
                desired_signal = 0.0
            # Exit if price hits BB upper in mean reversion
            if above_bb_upper and not strong_bullish:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if trend reverses strongly
            if strong_bullish:
                desired_signal = 0.0
            # Exit if RSI oversold (take profit)
            if rsi_oversold:
                desired_signal = 0.0
            # Exit if price hits BB lower in mean reversion
            if below_bb_lower and not strong_bearish:
                desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if 12h trend still bullish and RSI not extreme
                if trend_12h_bullish and rsi_1h[i] < 75:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if 12h trend still bearish and RSI not extreme
                if trend_12h_bearish and rsi_1h[i] > 25:
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
                # Flip position
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