#!/usr/bin/env python3
"""
Experiment #855: 1h Primary + 4h/1d HTF — Multi-Confluence with Relaxed Entries

Hypothesis: After 591+ failed strategies, the key insight is that 1h timeframe 
needs BALANCED entry conditions - strict enough to avoid fee drag, but relaxed 
enough to guarantee trades on ALL symbols (BTC, ETH, SOL).

Strategy design:
1. 1h Primary timeframe (target 40-80 trades/year)
2. 4h HMA(21) + 1d HMA(21) for trend bias (dual HTF confirmation)
3. 1h Fisher Transform(9) for reversal detection (proven in bear markets)
4. 1h RSI(14) with RELAXED thresholds (35/65, extremes 25/75)
5. 1h Choppiness Index(14) for regime detection
6. Volume confirmation (>0.8x 20-period avg)
7. Session filter (8-20 UTC) - but NOT required, just adds confidence
8. 1h ATR(14) for trailing stop (2.0x)
9. CRITICAL: Multiple entry triggers to guarantee trades on all symbols

Key changes from failed 1h strategies:
- RELAXED RSI thresholds (35/65 not 30/70) - more signals
- Multiple entry paths (Fisher cross, RSI extreme, Donchian breakout)
- Session filter adds confidence but NOT required for entry
- Extreme RSI alone triggers entry (guarantees trades during crashes/rallies)
- Hold logic maintains position through minor pullbacks

Why this should work:
- Dual HTF (4h+1d) reduces whipsaw vs single HTF
- Fisher Transform catches reversals in 2022 crash and 2025 bear market
- Relaxed thresholds ensure 40+ trades/year on ALL symbols
- Session filter reduces noise but doesn't block entries

Target: Sharpe > 0.612, trades >= 40 train, >= 5 test, ALL symbols positive
Timeframe: 1h (target 40-80 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_rsi_dual_htf_chop_vol_session_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_ema(series, period):
    """Exponential Moving Average."""
    return pd.Series(series).ewm(span=period, min_periods=period, adjust=False).mean().values

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

def calculate_fisher_transform(high, low, period=9):
    """Ehlers Fisher Transform."""
    n = len(high)
    fisher = np.full(n, np.nan)
    fisher_prev = np.full(n, np.nan)
    
    if n < period + 1:
        return fisher, fisher_prev
    
    for i in range(period, n):
        hl2 = (high[i] + low[i]) / 2.0
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        range_val = highest_high - lowest_low
        if range_val < 1e-10:
            fisher[i] = fisher[i-1] if i > 0 else 0.0
            fisher_prev[i] = fisher[i-1] if i > 1 else 0.0
            continue
        
        normalized = (hl2 - lowest_low) / range_val
        normalized = np.clip(normalized, 0.001, 0.999)
        
        fisher_val = 0.5 * np.log((1 + normalized) / (1 - normalized))
        
        if i > 0 and not np.isnan(fisher[i-1]):
            fisher[i] = 0.7 * fisher_val + 0.3 * fisher[i-1]
            fisher_prev[i] = fisher[i-1]
        else:
            fisher[i] = fisher_val
            fisher_prev[i] = fisher_val
    
    return fisher, fisher_prev

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index."""
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

def calculate_donchian(high, low, period=20):
    """Donchian Channels."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

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
    
    # Calculate 1h indicators
    rsi_1h = calculate_rsi(close, period=14)
    chop_1h = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    atr_1h = calculate_atr(high, low, close, period=14)
    ema_21_1h = calculate_ema(close, 21)
    ema_50_1h = calculate_ema(close, 50)
    fisher_1h, fisher_prev_1h = calculate_fisher_transform(high, low, period=9)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(rsi_1h[i]) or np.isnan(chop_1h[i]) or np.isnan(atr_1h[i]):
            continue
        if atr_1h[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(ema_21_1h[i]) or np.isnan(ema_50_1h[i]):
            continue
        if np.isnan(vol_avg_20[i]) or vol_avg_20[i] <= 1e-10:
            continue
        if np.isnan(fisher_1h[i]) or np.isnan(fisher_prev_1h[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # Extract hour from open_time for session filter
        hour = (open_time[i] // 3600000) % 24
        in_session = 8 <= hour <= 20
        
        # === TREND BIAS (4h + 1d HMA) ===
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # Strong bias when both agree
        strong_bullish = trend_4h_bullish and trend_1d_bullish
        strong_bearish = trend_4h_bearish and trend_1d_bearish
        
        # === REGIME (1h Choppiness) ===
        ranging_regime = chop_1h[i] > 55
        trending_regime = chop_1h[i] < 45
        
        # === RSI SIGNALS (Relaxed thresholds) ===
        rsi_oversold = rsi_1h[i] < 40
        rsi_overbought = rsi_1h[i] > 60
        rsi_extreme_oversold = rsi_1h[i] < 30
        rsi_extreme_overbought = rsi_1h[i] > 70
        rsi_neutral_bull = 40 <= rsi_1h[i] < 50
        rsi_neutral_bear = 50 < rsi_1h[i] <= 60
        
        # === FISHER TRANSFORM ===
        fisher_oversold = fisher_1h[i] < -1.0
        fisher_overbought = fisher_1h[i] > 1.0
        fisher_cross_up = fisher_prev_1h[i] < -1.0 and fisher_1h[i] >= -1.0
        fisher_cross_down = fisher_prev_1h[i] > 1.0 and fisher_1h[i] <= 1.0
        fisher_rising = fisher_1h[i] > fisher_prev_1h[i]
        fisher_falling = fisher_1h[i] < fisher_prev_1h[i]
        
        # === VOLUME ===
        volume_ok = volume[i] > 0.8 * vol_avg_20[i]
        
        # === DONCHIAN ===
        donchian_break_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        donchian_break_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === EMA ALIGNMENT ===
        ema_bullish = ema_21_1h[i] > ema_50_1h[i]
        ema_bearish = ema_21_1h[i] < ema_50_1h[i]
        
        desired_signal = 0.0
        
        # === LONG ENTRIES ===
        if strong_bullish or (trend_4h_bullish and ema_bullish):
            long_confidence = 0
            
            # Fisher cross + RSI oversold (primary)
            if fisher_cross_up and rsi_oversold:
                long_confidence += 2
            
            # RSI extreme oversold (guarantees trades)
            if rsi_extreme_oversold:
                long_confidence += 2
            
            # Fisher oversold + RSI neutral
            if fisher_oversold and rsi_neutral_bull:
                long_confidence += 1
            
            # Donchian breakout in trending regime
            if trending_regime and donchian_break_long:
                long_confidence += 1
            
            # Volume confirmation
            if volume_ok:
                long_confidence += 1
            
            # Session filter (adds confidence)
            if in_session:
                long_confidence += 0.5
            
            if long_confidence >= 2.5:
                desired_signal = BASE_SIZE
            elif long_confidence >= 2.0:
                desired_signal = REDUCED_SIZE
        
        # === SHORT ENTRIES ===
        if strong_bearish or (trend_4h_bearish and ema_bearish):
            short_confidence = 0
            
            # Fisher cross + RSI overbought (primary)
            if fisher_cross_down and rsi_overbought:
                short_confidence += 2
            
            # RSI extreme overbought (guarantees trades)
            if rsi_extreme_overbought:
                short_confidence += 2
            
            # Fisher overbought + RSI neutral
            if fisher_overbought and rsi_neutral_bear:
                short_confidence += 1
            
            # Donchian breakout in trending regime
            if trending_regime and donchian_break_short:
                short_confidence += 1
            
            # Volume confirmation
            if volume_ok:
                short_confidence += 1
            
            # Session filter
            if in_session:
                short_confidence += 0.5
            
            if short_confidence >= 2.5:
                desired_signal = -BASE_SIZE
            elif short_confidence >= 2.0:
                desired_signal = -REDUCED_SIZE
        
        # === RANGING REGIME MEAN REVERSION ===
        if ranging_regime and desired_signal == 0.0:
            # Long at lower bound
            if rsi_extreme_oversold and fisher_oversold:
                if volume_ok or in_session:
                    desired_signal = REDUCED_SIZE
            
            # Short at upper bound
            if rsi_extreme_overbought and fisher_overbought:
                if volume_ok or in_session:
                    desired_signal = -REDUCED_SIZE
        
        # === STOPLOSS CHECK ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.0 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                if (trend_4h_bullish or trend_1d_bullish) and fisher_1h[i] < 1.5:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                if (trend_4h_bearish or trend_1d_bearish) and fisher_1h[i] > -1.5:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            if trend_4h_bearish and trend_1d_bearish and fisher_1h[i] > 1.5:
                desired_signal = 0.0
            if ranging_regime and rsi_1h[i] > 75:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            if trend_4h_bullish and trend_1d_bullish and fisher_1h[i] < -1.5:
                desired_signal = 0.0
            if ranging_regime and rsi_1h[i] < 25:
                desired_signal = 0.0
        
        # === DISCRETIZE ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= 0.25 else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -0.25 else -REDUCED_SIZE
        
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