#!/usr/bin/env python3
"""
Experiment #814: 4h Primary + 1d/1w HTF — KAMA Trend + Fisher Reversals + Choppiness Regime

Hypothesis: After analyzing 554 failed strategies, key insight is that COMPLEXITY kills performance.
Previous strategies failed because:
1. Too many conflicting entry conditions (0 trades)
2. CRSI overfitting (works on train, fails on test)
3. Hold logic creates whipsaws (exit too early)
4. Regime switching too sensitive

New approach - SIMPLER and MORE ROBUST:
1. KAMA(21) for adaptive trend (responds to volatility, less lag than EMA)
2. Fisher Transform(9) for reversal signals (proven in literature for bear markets)
3. Choppiness Index(14) for regime detection (45/55 thresholds for cleaner separation)
4. Donchian(20) for breakout confirmation (standard period, less noise than 15)
5. RSI(14) with 30/70 thresholds (relaxed from 35/65 for more trades)
6. ATR(14) trailing stop at 2.5x (wider than 2.0x to avoid premature exits)
7. NO complex hold logic - either in position or flat
8. 1d HMA(21) + 1w HMA(21) for HTF trend bias (proven in #811)

Key differences from #811:
- KAMA instead of SMA200 (adaptive to volatility)
- Fisher Transform for reversal timing (catches bear market rallies)
- Simpler entry logic (fewer AND conditions = more trades)
- Wider ATR stop (2.5x vs 2.0x = fewer premature exits)
- No complex hold logic (reduces whipsaws)
- Cleaner regime thresholds (45/55 vs 40/50)

Target: Sharpe > 0.612, trades >= 30 train, >= 5 test, ALL symbols positive
Timeframe: 4h (target 25-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_fisher_chop_donchian_1d1w_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=21, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average.
    Adapts to market noise - moves fast in trends, slow in chop.
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < period + slow_period:
        return kama
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        signal = np.abs(close[i] - close[i - period])
        noise = np.sum(np.abs(np.diff(close[i-period:i+1])))
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 1.0
    
    # Calculate Smoothing Constant (SC)
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = er * (fast_sc - slow_sc) + slow_sc
    sc = np.clip(sc, 0, 1)
    
    # Calculate KAMA
    kama[period] = close[period]
    for i in range(period + 1, n):
        if not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = close[i]
    
    return kama

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform.
    Converts price to Gaussian distribution for clearer reversal signals.
    Long when Fisher crosses above -1.5, short when crosses below +1.5.
    """
    n = len(high)
    fisher = np.full(n, np.nan)
    fisher_signal = np.full(n, np.nan)
    
    if n < period:
        return fisher, fisher_signal
    
    for i in range(period - 1, n):
        # Calculate typical price
        hl2 = (high[i-period+1:i+1] + low[i-period+1:i+1]) / 2
        highest_hl2 = np.max(hl2)
        lowest_hl2 = np.min(hl2)
        
        if highest_hl2 == lowest_hl2:
            fisher[i] = 0.0
            fisher_signal[i] = 0.0
            continue
        
        # Normalize to -1 to +1
        x = (hl2[-1] - lowest_hl2) / (highest_hl2 - lowest_hl2)
        x = np.clip(x, 0.001, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1 + x) / (1 - x))
        
        # Signal line (1-period lag)
        if i > period - 1 and not np.isnan(fisher[i-1]):
            fisher_signal[i] = fisher[i-1]
        else:
            fisher_signal[i] = fisher[i]
    
    return fisher, fisher_signal

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

def calculate_donchian(high, low, period=20):
    """Donchian Channels — highest high and lowest low over period."""
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

def calculate_hma(series, period):
    """Hull Moving Average for HTF alignment."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate primary (4h) indicators
    kama_4h = calculate_kama(close, period=21)
    rsi_4h = calculate_rsi(close, period=14)
    chop_4h = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    atr_4h = calculate_atr(high, low, close, period=14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, period=9)
    
    # Calculate and align 1d HMA for intermediate trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for long-term trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
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
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(rsi_4h[i]) or np.isnan(chop_4h[i]) or np.isnan(atr_4h[i]):
            continue
        if np.isnan(kama_4h[i]) or np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            continue
        if atr_4h[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === LONG-TERM TREND BIAS (1w HTF HMA21) ===
        trend_1w_bullish = close[i] > hma_1w_aligned[i]
        trend_1w_bearish = close[i] < hma_1w_aligned[i]
        
        # === INTERMEDIATE TREND (1d HTF HMA21) ===
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # === ADAPTIVE TREND (4h KAMA21) ===
        kama_bullish = close[i] > kama_4h[i]
        kama_bearish = close[i] < kama_4h[i]
        
        # === REGIME DETECTION (4h Choppiness Index) ===
        ranging_regime = chop_4h[i] > 55
        trending_regime = chop_4h[i] < 45
        
        # === RSI SIGNALS (Relaxed for more trades) ===
        rsi_oversold = rsi_4h[i] < 30
        rsi_overbought = rsi_4h[i] > 70
        
        # === FISHER TRANSFORM REVERSALS ===
        fisher_long_signal = fisher[i] > -1.5 and fisher_signal[i] <= -1.5
        fisher_short_signal = fisher[i] < 1.5 and fisher_signal[i] >= 1.5
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        desired_signal = 0.0
        
        # === RANGING REGIME (CHOP > 55) — Mean Reversion with Fisher ===
        if ranging_regime:
            # Long: RSI oversold + Fisher reversal + any trend alignment
            if rsi_oversold and fisher_long_signal:
                if trend_1w_bullish or trend_1d_bullish or kama_bullish:
                    desired_signal = BASE_SIZE
                else:
                    desired_signal = REDUCED_SIZE  # Enter anyway with reduced size
            
            # Short: RSI overbought + Fisher reversal + any trend alignment
            if rsi_overbought and fisher_short_signal:
                if trend_1w_bearish or trend_1d_bearish or kama_bearish:
                    desired_signal = -BASE_SIZE
                else:
                    desired_signal = -REDUCED_SIZE  # Enter anyway with reduced size
        
        # === TRENDING REGIME (CHOP < 45) — Trend Following with Donchian ===
        elif trending_regime:
            # Long: KAMA bullish + Donchian breakout (relaxed: 1d/1w also ok)
            if kama_bullish and donchian_breakout_long:
                desired_signal = BASE_SIZE
            elif (trend_1w_bullish or trend_1d_bullish) and donchian_breakout_long:
                desired_signal = REDUCED_SIZE
            
            # Short: KAMA bearish + Donchian breakout
            if kama_bearish and donchian_breakout_short:
                desired_signal = -BASE_SIZE
            elif (trend_1w_bearish or trend_1d_bearish) and donchian_breakout_short:
                desired_signal = -REDUCED_SIZE
            
            # Pullback entries in trend (RSI + Fisher confluence)
            if kama_bullish and rsi_oversold and fisher_long_signal:
                if desired_signal == 0:
                    desired_signal = REDUCED_SIZE
            
            if kama_bearish and rsi_overbought and fisher_short_signal:
                if desired_signal == 0:
                    desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55) — Conservative ===
        else:
            # Only enter on strong confluence
            if rsi_oversold and fisher_long_signal and (trend_1w_bullish or trend_1d_bullish):
                desired_signal = REDUCED_SIZE
            
            if rsi_overbought and fisher_short_signal and (trend_1w_bearish or trend_1d_bearish):
                desired_signal = -REDUCED_SIZE
            
            # Donchian breakout with trend confirmation
            if donchian_breakout_long and kama_bullish:
                desired_signal = REDUCED_SIZE if desired_signal == 0 else desired_signal
            
            if donchian_breakout_short and kama_bearish:
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
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Flip position
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