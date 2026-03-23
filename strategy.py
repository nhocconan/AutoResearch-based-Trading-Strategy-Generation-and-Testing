#!/usr/bin/env python3
"""
Experiment #1025: 1h Primary + 4h/1d HTF — Simplified RSI Mean Reversion + HTF Trend

Hypothesis: After analyzing 743+ failed strategies, the key insight is:
1. 1h strategies fail due to EITHER zero trades (too strict) OR fee drag (too many)
2. The winning pattern from #1023 (Sharpe=0.291) was CRSI + Donchian + HTF HMA
3. For 1h: use 4h HMA for trend BIAS (not strict filter), 1h RSI for timing
4. RELAXED thresholds are CRITICAL: RSI<40 (not <10), RSI>60 (not >90)
5. NO session filter (killed trades in #1015, #1018)
6. NO choppiness index (too many conditions = 0 trades in #1014)

Why this should work:
- 4h HMA21 gives trend direction without being too restrictive
- 1h RSI(14) with relaxed thresholds (35/65) ensures 50-100 trades/year
- ATR(14) 2.5x stoploss protects from catastrophic moves
- Volume filter at 0.7x avg (not 0.8x) to avoid killing signals
- Discrete signal sizes (0.0, ±0.25, ±0.30) minimize fee churn

Critical lessons from failures:
- #1015, #1018, #1024: 0 trades = session/volume filters too strict
- #1014, #1020: Negative Sharpe = too many conflicting conditions
- #1023: Positive Sharpe = simpler CRSI + Donchian + HTF worked

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 1h (target 50-100 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_rsi_meanrev_4h1d_hma_vol_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    for i in range(period, n):
        if avg_loss[i] == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / (avg_loss[i] + 1e-10)
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

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

def calculate_volume_ratio(volume, period=20):
    """Volume ratio vs rolling average."""
    n = len(volume)
    vol_ratio = np.full(n, np.nan)
    
    if n < period:
        return vol_ratio
    
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    
    for i in range(period, n):
        if vol_avg[i] > 1e-10:
            vol_ratio[i] = volume[i] / vol_avg[i]
    
    return vol_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 4h HMA21 for medium-term trend
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA21 for long-term trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (1h) indicators
    rsi_1h = calculate_rsi(close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    vol_ratio_1h = calculate_volume_ratio(volume, period=20)
    
    # Price position relative to 4h HMA (normalized)
    hma_4h_pct = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(hma_4h_aligned[i]) and hma_4h_aligned[i] > 1e-10:
            hma_4h_pct[i] = (close[i] - hma_4h_aligned[i]) / hma_4h_aligned[i]
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    REDUCED_SIZE = 0.15
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(rsi_1h[i]) or np.isnan(atr_1h[i]) or atr_1h[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(vol_ratio_1h[i]) or np.isnan(hma_4h_pct[i]):
            continue
        
        # === HTF TREND BIAS (4h HMA21) ===
        # Relaxed: use as bias, not hard filter
        trend_bullish = hma_4h_pct[i] > -0.02  # Price within 2% of 4h HMA or above
        trend_bearish = hma_4h_pct[i] < 0.02   # Price within 2% of 4h HMA or below
        
        # === 1d HMA for strong trend confirmation ===
        strong_bull = close[i] > hma_1d_aligned[i]
        strong_bear = close[i] < hma_1d_aligned[i]
        
        # === RSI MEAN REVERSION SIGNALS (RELAXED) ===
        rsi_oversold = rsi_1h[i] < 40  # Relaxed from <10
        rsi_overbought = rsi_1h[i] > 60  # Relaxed from >90
        rsi_extreme_oversold = rsi_1h[i] < 30
        rsi_extreme_overbought = rsi_1h[i] > 70
        
        # === VOLUME FILTER (RELAXED) ===
        vol_ok = vol_ratio_1h[i] > 0.7  # Relaxed from >0.8
        
        # === ATR VOLATILITY CHECK ===
        atr_ok = atr_1h[i] > 0  # Always true if not nan
        
        desired_signal = 0.0
        
        # === LONG ENTRIES ===
        if trend_bullish and vol_ok:
            if rsi_extreme_oversold:
                # Strong oversold + trend bias = full size
                desired_signal = BASE_SIZE
            elif rsi_oversold and strong_bull:
                # Moderate oversold + strong bullish = reduced size
                desired_signal = REDUCED_SIZE
            elif rsi_1h[i] < 45 and hma_4h_pct[i] > 0.01:
                # RSI < 45 + price above 4h HMA by 1% = entry
                desired_signal = REDUCED_SIZE
        
        # === SHORT ENTRIES ===
        if trend_bearish and vol_ok:
            if rsi_extreme_overbought:
                # Strong overbought + trend bias = full size
                desired_signal = -BASE_SIZE
            elif rsi_overbought and strong_bear:
                # Moderate overbought + strong bearish = reduced size
                desired_signal = -REDUCED_SIZE
            elif rsi_1h[i] > 55 and hma_4h_pct[i] < -0.01:
                # RSI > 55 + price below 4h HMA by 1% = entry
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
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if RSI becomes overbought or trend reverses
            if rsi_1h[i] > 65 or (not trend_bullish and rsi_1h[i] > 50):
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if RSI becomes oversold or trend reverses
            if rsi_1h[i] < 35 or (not trend_bearish and rsi_1h[i] < 50):
                desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if trend still bullish and RSI not overbought
                if trend_bullish and rsi_1h[i] < 60:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if trend still bearish and RSI not oversold
                if trend_bearish and rsi_1h[i] > 40:
                    desired_signal = -BASE_SIZE
        
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