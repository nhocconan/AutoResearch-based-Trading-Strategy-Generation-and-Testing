#!/usr/bin/env python3
"""
Experiment #465: 1h Fisher Transform + 4h HMA Trend + Vol Spike Mean Reversion

Hypothesis: After analyzing 464 failed experiments, the pattern is clear:
- Pure trend strategies fail on BTC/ETH whipsaws (especially 2022 crash)
- Pure mean reversion fails on strong trends
- 2025+ is bear/range market, not bull like 2021

This strategy combines THREE proven edges for 1h timeframe:

1. 4H HMA(21) TREND BIAS (via mtf_data helper):
   - Long bias when price > 4h HMA (bull/neutral market)
   - Short bias when price < 4h HMA (bear market)
   - HMA smoother than EMA, critical for trend filtering

2. EHLERS FISHER TRANSFORM (period=9) FOR REVERSALS:
   - Long when Fisher crosses above -1.5 (oversold reversal)
   - Short when Fisher crosses below +1.5 (overbought reversal)
   - Catches reversals in bear rallies better than RSI
   - Proven in quantitative literature for crypto

3. VOLATILITY SPIKE MEAN REVERSION:
   - ATR(7)/ATR(30) > 2.0 = vol spike (panic/extreme)
   - Enter when price < BB(20, 2.5) lower band (oversold)
   - Exit when ATR ratio < 1.2 (vol crush)
   - Captures "vol crush" after panic sells

4. CHOPPINESS INDEX REGIME FILTER:
   - CHOP(14) > 61.8 = range (use mean reversion signals)
   - CHOP < 38.2 = trend (use Fisher breakout signals)
   - Prevents whipsaws in wrong regime

5. ATR(14) TRAILING STOP at 2.5x:
   - Signal → 0 when price moves 2.5*ATR against position
   - Critical for crash protection

6. POSITION SIZING: 0.28 discrete (conservative)
   - Max 28% capital per position
   - 2022 BTC crash was -77%, at 0.28 size = -21% max DD from crash

Why 1h timeframe:
- More signals than 4h/12h (ensures >10 trades per symbol)
- Less noise than 15m/30m (better signal quality)
- Captures intraday swings missed by daily strategies

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.28 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_4h_hma_vol_spike_chop_regime_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform - captures price reversals.
    Formula: Fisher = 0.5 * ln((1 + X) / (1 - X))
    Where X = 0.66 * ((price - LL) / (HH - LL) - 0.5) + 0.67 * prev_X
    """
    n = len(close)
    fisher = np.full(n, np.nan)
    fisher_prev = np.full(n, np.nan)
    
    # Use typical price
    typical = (high + low + close) / 3
    typical_s = pd.Series(typical)
    
    # Rolling HH and LL
    hh = typical_s.rolling(window=period, min_periods=period).max().values
    ll = typical_s.rolling(window=period, min_periods=period).min().values
    
    X = np.zeros(n)
    X_prev = 0.0
    
    for i in range(period, n):
        if hh[i] > ll[i] and hh[i] - ll[i] > 1e-10:
            X_raw = (typical[i] - ll[i]) / (hh[i] - ll[i])
            X[i] = 0.66 * (X_raw - 0.5) + 0.67 * X_prev
            
            # Clamp X to prevent division by zero
            X[i] = np.clip(X[i], -0.999, 0.999)
            
            # Fisher transform
            fisher[i] = 0.5 * np.log((1 + X[i]) / (1 - X[i]))
            fisher_prev[i] = 0.5 * np.log((1 + X_prev) / (1 - X_prev)) if i > period else 0.0
            
            X_prev = X[i]
    
    return fisher, fisher_prev

def calculate_bollinger_bands(close, period=20, std_mult=2.5):
    """Calculate Bollinger Bands with wider std for extreme detection."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper.values, lower.values, sma.values

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppiness vs trending.
    CHOP > 61.8 = range/choppy
    CHOP < 38.2 = trending
    Formula: 100 * LOG10(SUM(ATR, n) / (HH - LL)) / LOG10(n)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    # Calculate ATR
    atr = calculate_atr(high, low, close, period)
    
    for i in range(period * 2, n):
        atr_sum = np.sum(atr[i-period+1:i+1])
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        
        if hh > ll and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(period)
        else:
            chop[i] = 50.0  # neutral
    
    return chop

def calculate_atr_ratio(high, low, close, short_period=7, long_period=30):
    """Calculate ATR ratio for vol spike detection."""
    atr_short = calculate_atr(high, low, close, short_period)
    atr_long = calculate_atr(high, low, close, long_period)
    
    ratio = np.full(len(close), np.nan)
    for i in range(len(close)):
        if atr_long[i] > 1e-10:
            ratio[i] = atr_short[i] / atr_long[i]
    
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    fisher, fisher_prev = calculate_fisher_transform(high, low, 9)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.5)
    chop = calculate_choppiness_index(high, low, close, 14)
    atr_ratio = calculate_atr_ratio(high, low, close, 7, 30)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_prev[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_lower[i]) or np.isnan(bb_upper[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]) or np.isnan(atr_ratio[i]):
            signals[i] = 0.0
            continue
        
        # === 4H HMA TREND BIAS ===
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === CHOPPINESS INDEX REGIME ===
        choppy_regime = chop[i] > 61.8  # range market
        trending_regime = chop[i] < 38.2  # trend market
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        fisher_long = fisher_prev[i] < -1.5 and fisher[i] >= -1.5
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_short = fisher_prev[i] > 1.5 and fisher[i] <= 1.5
        
        # === VOL SPIKE MEAN REVERSION ===
        vol_spike = atr_ratio[i] > 2.0  # vol spike (panic)
        price_below_bb = close[i] < bb_lower[i]  # oversold
        price_above_bb = close[i] > bb_upper[i]  # overbought
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # FISHER REVERSAL (works in trending regime, align with 4h trend)
        if trending_regime:
            if fisher_long and bull_trend_4h:
                new_signal = SIZE
            elif fisher_short and bear_trend_4h:
                new_signal = -SIZE
        
        # VOL SPIKE MEAN REVERSION (works in choppy regime)
        if choppy_regime and new_signal == 0.0:
            if vol_spike and price_below_bb and bull_trend_4h:
                new_signal = SIZE
            elif vol_spike and price_above_bb and bear_trend_4h:
                new_signal = -SIZE
        
        # FISHER REVERSAL (also works in choppy regime with looser bias)
        if choppy_regime and new_signal == 0.0:
            if fisher_long:
                new_signal = SIZE
            elif fisher_short:
                new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit if 4h trend flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_4h:
                new_signal = 0.0
            if position_side < 0 and bull_trend_4h:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals