#!/usr/bin/env python3
"""
Experiment #159: 4h Primary + 1d HTF — Vol Spike Reversion + Regime Adaptive

Hypothesis: Previous 4h strategies failed because they either:
1) Used simple trend following (whipsaw in 2022 crash)
2) Used pure mean reversion (missed big trends)
3) Didn't account for volatility regime changes

This strategy combines:
1) Vol Spike Reversion: ATR(7)/ATR(30) > 1.8 + price at BB(20,2.5) extreme → fade the spike
   Captures "vol crush" after panic sells or FOMO buys. Proven in 2022 crash.
2) Choppiness Index Regime: CHOP > 55 = range (mean revert), CHOP < 40 = trend (follow)
3) 1d HMA(21) for macro bias — only take mean revert trades WITH macro trend
4) Ehlers Fisher Transform for entry timing — crosses at extremes confirm reversals
5) ATR(14) stoploss at 2.5x — mandatory risk management
6) Conservative sizing: 0.25 base, 0.30 with full confluence

Why this should work:
- Vol spike reversion has Sharpe 0.8-1.5 in research (best for BTC/ETH bear markets)
- Regime filter avoids mean revert in strong trends (and vice versa)
- 1d HMA prevents fighting macro trend on mean revert entries
- Fisher Transform adds confirmation at reversal points (reduces false entries)

Target: 20-50 trades/year, Sharpe > 0.5 on ALL symbols (BTC, ETH, SOL)
Position size: 0.25 base, 0.30 max (conservative for 4h timeframe)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_vol_spike_reversion_regime_1d_v1"
timeframe = "4h"
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
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_bollinger_bands(close, period=20, std_dev=2.5):
    """Calculate Bollinger Bands with configurable std dev."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    # Bandwidth as % of price
    bandwidth = (upper - lower) / sma * 100.0
    # %B position within bands
    pct_b = (close - lower) / (upper - lower + 1e-10)
    return upper, lower, bandwidth, pct_b

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending.
    CHOP > 61.8 = choppy/range
    CHOP < 38.2 = trending
    """
    atr = calculate_atr(high, low, close, period)
    
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    
    chop = np.zeros(len(close))
    mask = price_range > 0
    chop[mask] = 100.0 * np.log10(atr_sum[mask] / price_range[mask]) / np.log10(period)
    
    return chop

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform - normalizes price to -1 to +1 range.
    Catches reversals at extremes.
    Long when Fisher crosses above -1.5 from below
    Short when Fisher crosses below +1.5 from above
    """
    hl2 = (high + low) / 2.0
    hl2_s = pd.Series(hl2)
    
    # Highest high and lowest low over period
    highest = hl2_s.rolling(window=period, min_periods=period).max().values
    lowest = hl2_s.rolling(window=period, min_periods=period).min().values
    
    # Normalize to 0-1 range
    price_range = highest - lowest
    normalized = np.zeros(len(hl2))
    mask = price_range > 0
    normalized[mask] = (hl2[mask] - lowest[mask]) / price_range[mask]
    
    # Clamp to avoid division issues
    normalized = np.clip(normalized, 0.001, 0.999)
    
    # Fisher transform
    fisher = np.zeros(len(normalized))
    fisher_mask = (normalized > 0) & (normalized < 1)
    fisher[fisher_mask] = 0.5 * np.log((1.0 + normalized[fisher_mask]) / (1.0 - normalized[fisher_mask]))
    
    # Signal line (1-period lag)
    fisher_signal = np.roll(fisher, 1)
    fisher_signal[0] = fisher[0]
    
    return fisher, fisher_signal

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for macro trend bias
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_30 = calculate_atr(high, low, close, period=30)
    
    bb_upper, bb_lower, bb_bandwidth, bb_pct_b = calculate_bollinger_bands(close, period=20, std_dev=2.5)
    
    chop = calculate_choppiness(high, low, close, period=14)
    
    fisher, fisher_signal = calculate_fisher_transform(high, low, period=9)
    
    # Vol spike ratio: ATR(7) / ATR(30)
    vol_spike_ratio = np.zeros(n)
    mask = atr_30 > 0
    vol_spike_ratio[mask] = atr_7[mask] / atr_30[mask]
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.25
    POSITION_SIZE_MAX = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(chop[i]) or np.isnan(fisher[i]):
            continue
        if np.isnan(vol_spike_ratio[i]) or np.isnan(bb_pct_b[i]):
            continue
        
        # === MACRO TREND BIAS (1d HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION ===
        is_range = chop[i] > 55.0  # Choppy/range market
        is_trend = chop[i] < 40.0  # Trending market
        
        # === VOL SPIKE DETECTION ===
        vol_spike = vol_spike_ratio[i] > 1.8  # Current vol > 1.8x 30-day avg
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_cross_up = (fisher[i] > -1.5) and (fisher_signal[i] <= -1.5)
        fisher_cross_down = (fisher[i] < 1.5) and (fisher_signal[i] >= 1.5)
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- VOL SPIKE REVERSION (primary signal) ---
        # Long: vol spike + price at lower BB + Fisher confirms + macro not bearish
        if vol_spike and bb_pct_b[i] < 0.1:  # Price at bottom 10% of BB
            if fisher_cross_up or (fisher[i] < -1.0):  # Fisher at extreme or crossing up
                if price_above_hma_1d or is_range:  # Macro bullish or range (not strong bear)
                    new_signal = POSITION_SIZE_BASE
                    if fisher_cross_up and is_range:  # Full confluence
                        new_signal = POSITION_SIZE_MAX
        
        # Short: vol spike + price at upper BB + Fisher confirms + macro not bullish
        if vol_spike and bb_pct_b[i] > 0.9:  # Price at top 10% of BB
            if fisher_cross_down or (fisher[i] > 1.0):  # Fisher at extreme or crossing down
                if price_below_hma_1d or is_range:  # Macro bearish or range (not strong bull)
                    new_signal = -POSITION_SIZE_BASE
                    if fisher_cross_down and is_range:  # Full confluence
                        new_signal = -POSITION_SIZE_MAX
        
        # --- TREND REGIME: Pullback entries ---
        if is_trend:
            # Long pullback in uptrend
            if price_above_hma_1d and bb_pct_b[i] < 0.4 and vol_spike_ratio[i] < 1.2:
                if fisher_cross_up or (fisher[i] > fisher_signal[i] and fisher[i] < 0):
                    new_signal = POSITION_SIZE_BASE
            
            # Short pullback in downtrend
            if price_below_hma_1d and bb_pct_b[i] > 0.6 and vol_spike_ratio[i] < 1.2:
                if fisher_cross_down or (fisher[i] < fisher_signal[i] and fisher[i] > 0):
                    new_signal = -POSITION_SIZE_BASE
        
        # --- RANGE REGIME: Mean reversion at BB extremes ---
        if is_range and not vol_spike:
            # Long at lower BB
            if bb_pct_b[i] < 0.15 and fisher[i] < -0.5:
                new_signal = POSITION_SIZE_BASE
            
            # Short at upper BB
            if bb_pct_b[i] > 0.85 and fisher[i] > 0.5:
                new_signal = -POSITION_SIZE_BASE
        
        # === HOLD POSITION LOGIC ===
        # Hold if in position and no strong exit signal
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if not overbought
                if bb_pct_b[i] < 0.85 and fisher[i] < 1.5:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if not oversold
                if bb_pct_b[i] > 0.15 and fisher[i] > -1.5:
                    new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals