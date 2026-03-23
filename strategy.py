#!/usr/bin/env python3
"""
Experiment #734: 4h Primary + 12h/1d HTF — KAMA Adaptive Trend + Fisher Transform Reversals

Hypothesis: After 492 failed strategies, the key insight is that static indicators (EMA, RSI)
fail in changing regimes. KAMA (Kaufman Adaptive Moving Average) adapts its smoothing based
on market efficiency - smooth in choppy markets, responsive in trends. Combined with
Ehlers Fisher Transform (proven reversal detector in bear markets), this should catch
trends early while avoiding whipsaws.

Key innovations vs failed experiments:
1. KAMA instead of HMA/EMA - adapts to market regime automatically
2. Fisher Transform for reversals (not RSI) - better at catching bear market turns
3. 12h HMA for trend bias (not 1d/1w which are too slow for 4h entries)
4. Volume spike confirmation (2x avg) to filter false breakouts
5. Asymmetric sizing: 0.30 for trend, 0.20 for mean reversion

Target: Beat Sharpe=0.612, trades 25-50/year, ALL symbols positive Sharpe
Timeframe: 4h (proven working timeframe)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_fisher_volume_12h1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average.
    Adapts smoothing based on market efficiency (trend vs chop).
    ER = |change| / sum(|changes|) → 1 in trend, 0 in chop
    SC = (ER * (fast - slow) + slow)^2
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < period + slow_period:
        return kama
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(period, n):
        signal = np.abs(close[i] - close[i - period])
        noise = np.sum(np.abs(np.diff(close[max(0, i-period):i+1])))
        er[i] = signal / (noise + 1e-10) if noise > 0 else 0
    
    er = np.clip(er, 0, 1)
    
    # Calculate Smoothing Constant
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama[period] = close[period]
    for i in range(period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform.
    Normalizes price to Gaussian distribution for better reversal detection.
    Long when Fisher crosses above -1.5, Short when crosses below +1.5
    """
    n = len(high)
    fisher = np.full(n, np.nan)
    fisher_signal = np.full(n, np.nan)
    
    if n < period:
        return fisher, fisher_signal
    
    # Calculate typical price and normalize
    for i in range(period, n):
        hl2 = (high[i] + low[i]) / 2
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        if highest == lowest:
            continue
        
        # Normalize to -1 to +1
        normalized = 2 * ((hl2 - lowest) / (highest - lowest)) - 1
        normalized = np.clip(normalized, -0.999, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1 + normalized) / (1 - normalized))
        
        # Signal line (1-period lag)
        if i > period:
            fisher_signal[i] = fisher[i-1]
    
    return fisher, fisher_signal

def calculate_hma(series, period):
    """Hull Moving Average - smoother and more responsive than EMA."""
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
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_spike(volume, period=20, threshold=2.0):
    """Detect volume spikes (volume > threshold * avg volume)."""
    n = len(volume)
    spike = np.zeros(n, dtype=bool)
    
    if n < period:
        return spike
    
    avg_vol = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    
    for i in range(period, n):
        if avg_vol[i] > 0 and volume[i] > threshold * avg_vol[i]:
            spike[i] = True
    
    return spike

def calculate_sma(close, period):
    """Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (4h) indicators
    kama_4h = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    fisher_4h, fisher_signal_4h = calculate_fisher_transform(high, low, period=9)
    atr_4h = calculate_atr(high, low, close, period=14)
    sma_200 = calculate_sma(close, period=200)
    vol_spike_4h = calculate_volume_spike(volume, period=20, threshold=2.0)
    
    # Calculate and align HTF HMA for trend bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    TREND_SIZE = 0.30
    REVERSION_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(300, n):  # Need buffer for all indicators + HTF alignment
        # Skip if indicators not ready
        if np.isnan(kama_4h[i]) or np.isnan(atr_4h[i]) or atr_4h[i] <= 1e-10:
            continue
        if np.isnan(fisher_4h[i]) or np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(sma_200[i]):
            continue
        
        # === TREND BIAS (12h and 1d HTF HMA) ===
        trend_12h_bullish = close[i] > hma_12h_aligned[i]
        trend_12h_bearish = close[i] < hma_12h_aligned[i]
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # Strong trend when both agree
        strong_bullish = trend_12h_bullish and trend_1d_bullish
        strong_bearish = trend_12h_bearish and trend_1d_bearish
        
        # === KAMA TREND DIRECTION ===
        kama_bullish = close[i] > kama_4h[i]
        kama_bearish = close[i] < kama_4h[i]
        
        # KAMA slope (adaptive trend strength)
        kama_slope = 0.0
        if i > 5 and not np.isnan(kama_4h[i-5]):
            kama_slope = (kama_4h[i] - kama_4h[i-5]) / (kama_4h[i-5] + 1e-10)
        
        # === FISHER TRANSFORM REVERSAL SIGNALS ===
        fisher_long = False
        fisher_short = False
        
        if not np.isnan(fisher_signal_4h[i]):
            # Long: Fisher crosses above -1.5 from below
            if fisher_signal_4h[i] < -1.5 and fisher_4h[i] >= -1.5:
                fisher_long = True
            # Short: Fisher crosses below +1.5 from above
            if fisher_signal_4h[i] > 1.5 and fisher_4h[i] <= 1.5:
                fisher_short = True
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        desired_signal = 0.0
        current_size = TREND_SIZE
        
        # === LONG ENTRY CONDITIONS (multiple paths) ===
        long_signal = False
        
        # Path 1: Strong bullish trend + KAMA bullish + Fisher reversal
        if strong_bullish and kama_bullish and fisher_long:
            long_signal = True
            current_size = TREND_SIZE
        
        # Path 2: KAMA bullish + above SMA200 + volume spike confirmation
        if kama_bullish and above_sma200 and vol_spike_4h[i] and kama_slope > 0.001:
            long_signal = True
            current_size = TREND_SIZE
        
        # Path 3: Mean reversion - Fisher long + price below KAMA (oversold in uptrend)
        if fisher_long and trend_12h_bullish and close[i] < kama_4h[i]:
            long_signal = True
            current_size = REVERSION_SIZE
        
        # Path 4: Trend continuation - KAMA bullish + 12h bullish + RSI-like Fisher neutral
        if kama_bullish and trend_12h_bullish and -1.0 < fisher_4h[i] < 1.0:
            long_signal = True
            current_size = TREND_SIZE
        
        if long_signal:
            desired_signal = current_size
        
        # === SHORT ENTRY CONDITIONS (multiple paths) ===
        short_signal = False
        
        # Path 1: Strong bearish trend + KAMA bearish + Fisher reversal
        if strong_bearish and kama_bearish and fisher_short:
            short_signal = True
            current_size = TREND_SIZE
        
        # Path 2: KAMA bearish + below SMA200 + volume spike confirmation
        if kama_bearish and below_sma200 and vol_spike_4h[i] and kama_slope < -0.001:
            short_signal = True
            current_size = TREND_SIZE
        
        # Path 3: Mean reversion - Fisher short + price above KAMA (overbought in downtrend)
        if fisher_short and trend_12h_bearish and close[i] > kama_4h[i]:
            short_signal = True
            current_size = REVERSION_SIZE
        
        # Path 4: Trend continuation - KAMA bearish + 12h bearish + Fisher neutral
        if kama_bearish and trend_12h_bearish and -1.0 < fisher_4h[i] < 1.0:
            short_signal = True
            current_size = TREND_SIZE
        
        if short_signal:
            desired_signal = -current_size
        
        # === CONFLICT RESOLUTION ===
        if long_signal and short_signal:
            # Go with stronger HTF trend (1d HMA)
            if trend_1d_bullish:
                desired_signal = TREND_SIZE
            elif trend_1d_bearish:
                desired_signal = -TREND_SIZE
            else:
                desired_signal = 0.0
        
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
                # Hold long if KAMA still bullish and Fisher not extremely overbought
                if kama_bullish and fisher_4h[i] < 2.0:
                    desired_signal = TREND_SIZE
            elif position_side < 0:
                # Hold short if KAMA still bearish and Fisher not extremely oversold
                if kama_bearish and fisher_4h[i] > -2.0:
                    desired_signal = -TREND_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if KAMA reverses or Fisher extremely overbought
            if kama_bearish or fisher_4h[i] > 2.5:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if KAMA reverses or Fisher extremely oversold
            if kama_bullish or fisher_4h[i] < -2.5:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = TREND_SIZE if desired_signal >= TREND_SIZE else REVERSION_SIZE
        elif desired_signal < 0:
            desired_signal = -TREND_SIZE if desired_signal <= -TREND_SIZE else -REVERSION_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
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