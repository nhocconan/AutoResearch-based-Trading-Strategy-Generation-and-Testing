#!/usr/bin/env python3
"""
Experiment #275: 6h Primary + 1d/1w HTF — Fisher Transform + Asymmetric Trend v1

Hypothesis: 6h timeframe needs SIMPLER logic than 12h/4h. Previous 6h strategies failed
because they used overly complex regime detection (CHOP, Fisher+CHOP, etc.). This strategy:

1. EHLERS FISHER TRANSFORM (period=9): Proven reversal indicator that normalizes price
   into -1.5 to +1.5 range. Long when Fisher crosses above -1.0, short when crosses below +1.0.
   
2. ASYMMETRIC TREND FILTER: Only trade LONG when 1w HMA bullish, only trade SHORT when
   1w HMA bearish. This avoids fighting the major trend (key lesson from 2022 crash).
   
3. 1d HMA CONFIRMATION: Intermediate trend must align with entry direction (1d HMA).
   
4. VOLATILITY FILTER: ATR(14)/ATR(50) > 0.7 ensures we're not trading in extreme compression.
   
5. RSI FILTER: RSI(14) between 35-65 for entries (avoid extreme overbought/oversold).

6. DISCRETE SIZING: 0.30 with 1w confirmation, 0.20 without. Stoploss at 2.5x ATR.

Why this might work on 6h:
- Fisher Transform catches reversals better than RSI in bear/range markets
- Asymmetric entries reduce whipsaw (only trade with 1w trend)
- Simpler than CHOP-based regime detection (which failed in #267, #272)
- 6h has fewer bars than 4h/1h, so fewer false signals

Target: Sharpe>0.40 (beat current best 0.399), DD>-40%, trades>=20 train, trades>=3 test
Timeframe: 6h | HTF: 1d, 1w | Size: 0.20-0.30 (discrete)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_asymmetric_trend_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform - normalizes price into Gaussian distribution
    Range: approximately -1.5 to +1.5
    Long signal: Fisher crosses above -1.0 from below
    Short signal: Fisher crosses below +1.0 from above
    
    Formula:
    1. Calculate typical price: (High + Low) / 2
    2. Normalize: (price - lowest) / (highest - lowest) * 2 - 1
    3. Apply Fisher: 0.5 * ln((1 + x) / (1 - x))
    4. Smooth with EMA
    """
    n = len(high)
    if n < period + 5:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    # Typical price
    typical = (high + low) / 2.0
    
    # Normalize price over lookback period
    normalized = np.zeros(n)
    normalized[:] = np.nan
    
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        price_range = highest - lowest
        
        if price_range > 1e-10:
            normalized[i] = 2.0 * (typical[i] - lowest) / price_range - 1.0
            # Clamp to avoid division by zero in Fisher
            normalized[i] = max(-0.999, min(0.999, normalized[i]))
    
    # Fisher transform
    fisher = np.zeros(n)
    fisher[:] = np.nan
    
    for i in range(period, n):
        if not np.isnan(normalized[i]):
            x = normalized[i]
            fisher[i] = 0.5 * np.log((1.0 + x) / (1.0 - x))
    
    # Smooth Fisher with EMA
    fisher_smooth = pd.Series(fisher).ewm(span=3, min_periods=3, adjust=False).mean().values
    
    return fisher_smooth, fisher

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=34)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (6h) indicators
    hma_6h = calculate_hma(close, period=21)
    atr = calculate_atr(high, low, close, period=14)
    atr_50 = calculate_atr(high, low, close, period=50)
    rsi = calculate_rsi(close, period=14)
    fisher_smooth, fisher_raw = calculate_fisher_transform(high, low, period=9)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # Track Fisher crosses
    prev_fisher = np.nan
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(atr_50[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            prev_fisher = fisher_smooth[i] if not np.isnan(fisher_smooth[i]) else prev_fisher
            continue
        
        if np.isnan(hma_6h[i]) or np.isnan(rsi[i]) or np.isnan(fisher_smooth[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            prev_fisher = fisher_smooth[i] if not np.isnan(fisher_smooth[i]) else prev_fisher
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            prev_fisher = fisher_smooth[i] if not np.isnan(fisher_smooth[i]) else prev_fisher
            continue
        
        # === VOLATILITY FILTER ===
        # Only trade when not in extreme compression
        vol_ratio = atr[i] / atr_50[i] if atr_50[i] > 1e-10 else 0.0
        vol_ok = vol_ratio > 0.7
        
        # === HTF TREND BIAS (ASYMMETRIC) ===
        # 1w HMA determines major trend direction
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # 1d HMA for intermediate confirmation
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === 6h HMA TREND ===
        hma_bull = close[i] > hma_6h[i]
        hma_bear = close[i] < hma_6h[i]
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === RSI FILTER (avoid extremes) ===
        rsi_ok_long = 35.0 <= rsi[i] <= 70.0
        rsi_ok_short = 30.0 <= rsi[i] <= 65.0
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_long = False
        fisher_short = False
        
        if not np.isnan(prev_fisher) and not np.isnan(fisher_smooth[i]):
            # Long: Fisher crosses above -1.0 from below
            if prev_fisher <= -1.0 and fisher_smooth[i] > -1.0:
                fisher_long = True
            # Short: Fisher crosses below +1.0 from above
            if prev_fisher >= 1.0 and fisher_smooth[i] < 1.0:
                fisher_short = True
        
        prev_fisher = fisher_smooth[i]
        
        # === ENTRY LOGIC (ASYMMETRIC) ===
        desired_signal = 0.0
        
        # LONG ENTRY: Only when 1w bullish (asymmetric)
        if htf_1w_bull and vol_ok and rsi_ok_long:
            # Strong signal: 1w bull + 1d bull + Fisher long + 6h HMA bull
            if fisher_long and htf_1d_bull and hma_bull and above_sma200:
                desired_signal = SIZE_STRONG
            # Base signal: 1w bull + Fisher long + 6h HMA bull
            elif fisher_long and hma_bull and above_sma200:
                desired_signal = SIZE_BASE
        
        # SHORT ENTRY: Only when 1w bearish (asymmetric)
        elif htf_1w_bear and vol_ok and rsi_ok_short:
            # Strong signal: 1w bear + 1d bear + Fisher short + 6h HMA bear
            if fisher_short and htf_1d_bear and hma_bear and below_sma200:
                desired_signal = -SIZE_STRONG
            # Base signal: 1w bear + Fisher short + 6h HMA bear
            elif fisher_short and hma_bear and below_sma200:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i] if position_side > 0 else 0.0
                lowest_since_entry = low[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i] if position_side > 0 else 0.0
                lowest_since_entry = low[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, low[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals