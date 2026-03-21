#!/usr/bin/env python3
"""
Experiment #282: 1d Fisher Transform Reversals with HMA Trend & Choppiness Regime Filter
Hypothesis: Daily timeframe reduces noise and whipsaws. Fisher Transform excels at catching
reversals in bear markets (2022 crash bottom, 2025 bear rallies). Choppiness Index filters
out range-bound periods where trend strategies fail. Weekly HMA provides macro bias to avoid
counter-trend trades. This combines mean-reversion (Fisher) with trend-following (HMA) in a
regime-adaptive framework. Position sizing: 0.25 entry, 0.125 half at 2R. Stoploss: 2.5*ATR.
Target: Beat Sharpe=0.499 from current best by reducing whipsaws in 2022/2025 bear markets.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_fisher_hma_chop_weekly_regime_atr_v1"
timeframe = "1d"
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
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_fisher_transform(close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution.
    Catches reversals at extreme values (+/- 2 sigma).
    """
    close_s = pd.Series(close)
    # Calculate highest high and lowest low over period
    hh = close_s.rolling(window=period, min_periods=period).max().values
    ll = close_s.rolling(window=period, min_periods=period).min().values
    # Normalize to -1 to +1 range
    range_hl = hh - ll
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    normalized = 0.66 * ((close - ll) / range_hl - 0.5) + 0.67 * np.roll(normalized, 1) if len(close) > 1 else np.zeros(len(close))
    # Recalculate properly without recursion
    normalized = np.zeros(len(close))
    for i in range(period, len(close)):
        norm_val = 0.66 * ((close[i] - ll[i]) / range_hl[i] - 0.5)
        normalized[i] = norm_val + 0.67 * normalized[i-1]
    normalized = np.clip(normalized, -0.99, 0.99)
    # Fisher transform
    fisher = 0.5 * np.log((1 + normalized) / (1 - normalized))
    fisher_signal = np.roll(fisher, 1)
    fisher_signal[0] = fisher[0]
    return fisher, fisher_signal

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppiness vs trending.
    CHOP > 61.8 = range-bound (mean reversion regime)
    CHOP < 38.2 = trending (trend following regime)
    """
    atr = calculate_atr(high, low, close, period)
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    range_hl = hh - ll
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    chop = 100 * np.log10(atr_sum / (period * range_hl)) / np.log10(period)
    chop = np.nan_to_num(chop, nan=50.0)
    return chop

def calculate_hma_slope(hma, lookback=3):
    """Calculate HMA slope direction."""
    slope = np.zeros(len(hma))
    for i in range(lookback, len(hma)):
        if hma[i] > hma[i-lookback]:
            slope[i] = 1.0
        elif hma[i] < hma[i-lookback]:
            slope[i] = -1.0
        else:
            slope[i] = 0.0
    return slope

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    hma_1d = calculate_hma(close, 21)
    hma_1d_slope = calculate_hma_slope(hma_1d, 3)
    fisher, fisher_signal = calculate_fisher_transform(close, 9)
    chop = calculate_choppiness_index(high, low, close, 14)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.125
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # HTF trend filter (weekly macro bias)
        weekly_bullish = close[i] > hma_1w_aligned[i]
        weekly_bearish = close[i] < hma_1w_aligned[i]
        
        # 1d trend direction
        daily_uptrend = hma_1d_slope[i] > 0
        daily_downtrend = hma_1d_slope[i] < 0
        
        # Regime filter (choppiness)
        trending_regime = chop[i] < 50  # More lenient than 38.2 to get more trades
        ranging_regime = chop[i] > 55   # More lenient than 61.8
        
        # Fisher Transform signals (reversal detection)
        fisher_long = fisher_signal[i] < -1.5 and fisher[i] > fisher_signal[i]
        fisher_short = fisher_signal[i] > 1.5 and fisher[i] < fisher_signal[i]
        fisher_extreme_long = fisher[i] < -2.0
        fisher_extreme_short = fisher[i] > 2.0
        
        # Price vs HMA for confirmation
        price_above_hma = close[i] > hma_1d[i]
        price_below_hma = close[i] < hma_1d[i]
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Fisher reversal + weekly bullish bias + trending regime
        if fisher_long and weekly_bullish:
            if trending_regime or price_above_hma:
                new_signal = SIZE_ENTRY
        
        # Fisher extreme oversold + daily uptrend
        elif fisher_extreme_long and daily_uptrend:
            new_signal = SIZE_ENTRY
        
        # Fisher reversal + ranging regime (mean reversion)
        elif fisher_extreme_long and ranging_regime:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRY ===
        # Fisher reversal + weekly bearish bias + trending regime
        if fisher_short and weekly_bearish:
            if trending_regime or price_below_hma:
                new_signal = -SIZE_ENTRY
        
        # Fisher extreme overbought + daily downtrend
        elif fisher_extreme_short and daily_downtrend:
            new_signal = -SIZE_ENTRY
        
        # Fisher reversal + ranging regime (mean reversion)
        elif fisher_extreme_short and ranging_regime:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from highest)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from lowest)
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = entry_price - close[i]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reduced (take profit)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals