#!/usr/bin/env python3
"""
Experiment #294: 4h Primary + 12h/1d HTF — Fisher Transform Reversal + HMA Trend

Hypothesis: Previous 4h regime-switching strategies failed (#284, #289, #291) because
Connors RSI + Choppiness was over-complicated. This version uses SIMPLER logic:
- 12h HMA(21) for MACRO trend bias (soft filter)
- 4h HMA(16/48) for PRIMARY trend direction
- Fisher Transform(9) for ENTRY timing (catches reversals better than RSI in bear markets)
- Choppiness Index(14) as regime filter (only trend-follow when CHOP < 50)
- ATR(14) 2.5x trailing stoploss
- Position size: 0.28 (conservative for 4h volatility)

KEY DIFFERENCE from failed attempts:
- Fisher Transform instead of RSI/CRSI (proven reversal catcher in Ehlers literature)
- Choppiness as FILTER only (not primary signal generator)
- Simpler entry: Fisher cross + HMA alignment + regime check
- TARGET: 25-45 trades/year on 4h, Sharpe > 0.5 on ALL symbols

Fisher Transform logic (Ehlers):
- Normalizes price to Gaussian distribution
- Long when Fisher crosses above -1.5 (oversold reversal)
- Short when Fisher crosses below +1.5 (overbought reversal)
- Works well in bear/range markets where RSI fails
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_hma_chop_regime_12h1d_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average (HMA)."""
    close_s = pd.Series(close)
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, period)
    hull = 2 * wma_half - wma_full
    hma = wma(hull, sqrt_n)
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Normalizes price to Gaussian distribution for better reversal detection.
    """
    hl2 = (high + low) / 2.0
    hl2_s = pd.Series(hl2)
    
    # Calculate highest high and lowest low over period
    hh = hl2_s.rolling(window=period, min_periods=period).max()
    ll = hl2_s.rolling(window=period, min_periods=period).min()
    
    # Normalize to 0-1 range
    with np.errstate(divide='ignore', invalid='ignore'):
        norm = (hl2 - ll) / (hh - ll + 1e-10)
    
    # Clamp to avoid division issues
    norm = np.clip(norm, 0.001, 0.999)
    
    # Fisher transform
    fisher = 0.5 * np.log((norm) / (1.0 - norm + 1e-10))
    fisher_s = pd.Series(fisher)
    
    # Signal line (1-period lag)
    fisher_signal = fisher_s.shift(1).values
    
    return fisher_s.values, fisher_signal

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    Values > 61.8 = choppy/range, < 38.2 = trending.
    We use 50 as threshold for simplicity.
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # Highest high and lowest low over period
    hh = high_s.rolling(window=period, min_periods=period).max()
    ll = low_s.rolling(window=period, min_periods=period).min()
    
    # Sum of ATR over period
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    tr_s = pd.Series(tr)
    atr_sum = tr_s.rolling(window=period, min_periods=period).sum()
    
    # CHOP formula
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100.0 * np.log10((hh - ll).values / (atr_sum.values + 1e-10)) / np.log10(period)
    
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0.0, 100.0)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h indicators (primary timeframe)
    hma_16 = calculate_hma(close, 16)
    hma_48 = calculate_hma(close, 48)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness_index(high, low, close, period=14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, period=9)
    
    # Calculate and align 12h HMA for medium-term bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate and align 1d HMA for macro bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.28  # Conservative for 4h volatility
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(hma_16[i]) or np.isnan(hma_48[i]):
            signals[i] = 0.0
            continue
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            signals[i] = 0.0
            continue
        if np.isnan(chop_14[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO BIAS (1d HMA) - SOFT FILTER ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === MEDIUM-TERM BIAS (12h HMA) ===
        price_above_hma_12h = close[i] > hma_12h_aligned[i]
        price_below_hma_12h = close[i] < hma_12h_aligned[i]
        
        # === 4h TREND (HMA crossover) - PRIMARY FILTER ===
        hma_4h_bullish = hma_16[i] > hma_48[i]
        hma_4h_bearish = hma_16[i] < hma_48[i]
        
        # === REGIME FILTER (Choppiness Index) ===
        # CHOP < 50 = trending regime (allow trend-following entries)
        # CHOP >= 50 = choppy regime (reduce position or skip)
        is_trending_regime = chop_14[i] < 50.0
        
        # === FISHER TRANSFORM ENTRY SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_long_signal = (fisher_signal[i] < -1.5) and (fisher[i] >= -1.5)
        fisher_short_signal = (fisher_signal[i] > 1.5) and (fisher[i] <= 1.5)
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRY: 4h bullish + Fisher long + trending regime + 12h bias
        if hma_4h_bullish and fisher_long_signal and is_trending_regime:
            # Soft confirmation: price above 12h HMA preferred but not required
            if price_above_hma_12h or price_above_hma_1d:
                desired_signal = POSITION_SIZE
        
        # SHORT ENTRY: 4h bearish + Fisher short + trending regime + 12h bias
        elif hma_4h_bearish and fisher_short_signal and is_trending_regime:
            # Soft confirmation: price below 12h HMA preferred but not required
            if price_below_hma_12h or price_below_hma_1d:
                desired_signal = -POSITION_SIZE
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        if in_position and position_side > 0 and hma_4h_bearish:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and hma_4h_bullish:
            desired_signal = 0.0
        
        # === FISHER EXTREME EXIT (take profit) ===
        # Exit long when Fisher > +1.5 (overbought)
        # Exit short when Fisher < -1.5 (oversold)
        if in_position and position_side > 0 and fisher[i] > 1.5:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and fisher[i] < -1.5:
            desired_signal = 0.0
        
        # === REGIME CHANGE EXIT ===
        # If regime becomes choppy while in position, reduce or exit
        if in_position and not is_trending_regime:
            # Reduce to half position in choppy regime
            if position_side > 0:
                desired_signal = POSITION_SIZE / 2
            elif position_side < 0:
                desired_signal = -POSITION_SIZE / 2
        
        # === HOLD LOGIC ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and hma_4h_bullish:
                desired_signal = POSITION_SIZE
            elif position_side < 0 and hma_4h_bearish:
                desired_signal = -POSITION_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else float('inf')
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else float('inf')
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals