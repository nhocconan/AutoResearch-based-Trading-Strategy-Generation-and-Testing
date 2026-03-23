#!/usr/bin/env python3
"""
Experiment #998: 30m Primary + 4h/1d HTF — Simplified Regime + RSI Pullback + Volume

Hypothesis: After 12 consecutive failures (many with 0 trades), the key is SIMPLER logic
with RELAXED entry conditions that actually trigger. Previous 30m strategies (#988, #995)
had Sharpe=0.000 because entry filters were too strict.

Key changes from failures:
1. REMOVED funding rate dependency (caused alignment issues, 0 trades)
2. SIMPLIFIED regime detection (CHOP > 50 = range, < 50 = trend)
3. RELAXED RSI thresholds (30/70 not 25/75) to ensure trades trigger
4. ADDED session filter (8-20 UTC) for liquidity
5. Volume filter is lenient (> 0.5x avg, not > 0.8x)
6. Hold logic maintains positions through minor pullbacks

Why 30m timeframe:
- Target 40-80 trades/year (balance between fee drag and opportunity)
- 4h HMA provides trend bias, 30m RSI provides entry timing
- Session filter reduces noise during low-liquidity hours
- Proven pattern: HTF trend + LTF pullback entries

Critical for trade generation:
- RSI < 35 OR RSI > 65 triggers (not extreme 25/75)
- Either trend confluence OR BB extreme triggers (not both required)
- Reduced size (0.20) for entries with less confluence
- Base size (0.30) for entries with full confluence

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 30m (target 50-80 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_simplified_regime_rsi_4h1d_hma_session_vol_v1"
timeframe = "30m"
leverage = 1.0

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

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands."""
    n = len(close)
    middle = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    bandwidth = (upper - lower) / (middle + 1e-10)
    
    return middle, upper, lower, bandwidth

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

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index — measures market choppy vs trending."""
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

def calculate_volume_ratio(volume, period=20):
    """Volume ratio: current volume / rolling avg volume."""
    n = len(volume)
    ratio = np.full(n, np.nan)
    
    avg_vol = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    
    for i in range(period - 1, n):
        if avg_vol[i] > 1e-10:
            ratio[i] = volume[i] / avg_vol[i]
    
    return ratio

def get_hour_from_timestamp(timestamps):
    """Extract hour from open_time timestamps (milliseconds)."""
    hours = np.zeros(len(timestamps), dtype=int)
    for i, ts in enumerate(timestamps):
        # Convert ms to seconds, then to datetime
        hours[i] = (ts // 1000 // 3600) % 24
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
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (30m) indicators
    rsi_30m = calculate_rsi(close, period=14)
    atr_30m = calculate_atr(high, low, close, period=14)
    bb_mid, bb_upper, bb_lower, bb_bw = calculate_bollinger(close, period=20, std_mult=2.0)
    chop_30m = calculate_choppiness(high, low, close, period=14)
    vol_ratio_30m = calculate_volume_ratio(volume, period=20)
    hours = get_hour_from_timestamps(open_time)
    
    # Calculate and align 4h HMA for medium-term trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for macro regime
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(rsi_30m[i]) or np.isnan(atr_30m[i]) or atr_30m[i] <= 1e-10:
            continue
        if np.isnan(bb_mid[i]) or np.isnan(chop_30m[i]):
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        in_session = 8 <= hours[i] <= 20
        
        # === VOLUME FILTER (lenient) ===
        volume_ok = vol_ratio_30m[i] > 0.5 if not np.isnan(vol_ratio_30m[i]) else True
        
        # === MACRO REGIME (1d HTF HMA21) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === MEDIUM-TERM TREND (4h HTF HMA21) ===
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # === REGIME DETECTION (30m Choppiness Index) ===
        ranging_regime = chop_30m[i] > 50
        trending_regime = chop_30m[i] < 50
        
        # === BOLLINGER BAND POSITION ===
        bb_range = bb_upper[i] - bb_lower[i]
        bb_position = (close[i] - bb_lower[i]) / bb_range if bb_range > 1e-10 else 0.5
        bb_lower_break = close[i] < bb_lower[i]
        bb_upper_break = close[i] > bb_upper[i]
        bb_extreme_low = bb_position < 0.15
        bb_extreme_high = bb_position > 0.85
        
        # === RSI SIGNALS (relaxed thresholds for trade generation) ===
        rsi_oversold = rsi_30m[i] < 35
        rsi_overbought = rsi_30m[i] > 65
        rsi_extreme_oversold = rsi_30m[i] < 25
        rsi_extreme_overbought = rsi_30m[i] > 75
        rsi_neutral = 35 <= rsi_30m[i] <= 65
        
        desired_signal = 0.0
        confluence_count = 0
        
        # === RANGING REGIME (CHOP > 50) — Mean Reversion ===
        if ranging_regime:
            # Long signals
            long_signals = 0
            if bb_lower_break:
                long_signals += 1
            if rsi_oversold:
                long_signals += 1
            if bb_extreme_low:
                long_signals += 1
            if macro_bull or trend_4h_bullish:
                long_signals += 1
            
            # Short signals
            short_signals = 0
            if bb_upper_break:
                short_signals += 1
            if rsi_overbought:
                short_signals += 1
            if bb_extreme_high:
                short_signals += 1
            if macro_bear or trend_4h_bearish:
                short_signals += 1
            
            # Enter long if 2+ signals
            if long_signals >= 2:
                desired_signal = BASE_SIZE if long_signals >= 3 else REDUCED_SIZE
            # Enter short if 2+ signals
            elif short_signals >= 2:
                desired_signal = -BASE_SIZE if short_signals >= 3 else -REDUCED_SIZE
            # Single signal with session/volume confirmation
            elif bb_extreme_low and in_session:
                desired_signal = REDUCED_SIZE
            elif bb_extreme_high and in_session:
                desired_signal = -REDUCED_SIZE
        
        # === TRENDING REGIME (CHOP < 50) — Trend Following ===
        elif trending_regime:
            # Long in bullish trend on pullback
            if macro_bull or trend_4h_bullish:
                if rsi_oversold or bb_lower_break:
                    desired_signal = BASE_SIZE if (rsi_oversold and bb_lower_break) else REDUCED_SIZE
                elif rsi_30m[i] < 45 and in_session:
                    desired_signal = REDUCED_SIZE
            
            # Short in bearish trend on rally
            if macro_bear or trend_4h_bearish:
                if rsi_overbought or bb_upper_break:
                    desired_signal = -BASE_SIZE if (rsi_overbought and bb_upper_break) else -REDUCED_SIZE
                elif rsi_30m[i] > 55 and in_session:
                    desired_signal = -REDUCED_SIZE
        
        # === GUARANTEED TRADE GENERATION (fallback) ===
        # If no signal yet, use simple RSI extremes to ensure trades
        if desired_signal == 0.0:
            if rsi_extreme_oversold and in_session:
                desired_signal = REDUCED_SIZE
            elif rsi_extreme_overbought and in_session:
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
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if trend intact and RSI not overbought
                if (macro_bull or trend_4h_bullish) and rsi_30m[i] < 70:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if trend intact and RSI not oversold
                if (macro_bear or trend_4h_bearish) and rsi_30m[i] > 30:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if RSI very overbought
            if rsi_30m[i] > 80:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if RSI very oversold
            if rsi_30m[i] < 20:
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
                entry_atr = atr_30m[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_30m[i]
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