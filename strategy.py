#!/usr/bin/env python3
"""
Experiment #520: 4h Fisher Transform with Dual HTF HMA + Choppiness Regime

Hypothesis: After 500+ failed experiments, the pattern is clear - simple trend 
following fails on 4h due to whipsaws, and pure mean-reversion fails in strong 
trends. The solution is REGIME-ADAPTIVE entries using Fisher Transform for 
reversal timing, combined with DUAL HTF confirmation (1d + 1w HMA) for trend 
bias, and CHOPPINESS INDEX to distinguish trending vs ranging markets.

Key innovations:
1. FISHER TRANSFORM (period=9): Normalizes price to Gaussian distribution, 
   extreme values indicate reversal points. Long when Fisher crosses above -1.5,
   short when crosses below +1.5.
2. DUAL HTF HMA: 1d HMA(21) for primary trend, 1w HMA(21) for macro bias.
   Only long when both HTF HMAs agree bullish, only short when both bearish.
3. CHOPPINESS INDEX (14): CHOP > 61.8 = range (use Fisher reversals),
   CHOP < 38.2 = trend (use pullback entries to EMA21).
4. ASYMMETRIC SIZING: 0.35 in confirmed trends, 0.25 in ranges (less risk in chop)
5. LOOSE ENTRY THRESHOLDS: Fisher > -1.8 ensures >=10 trades/year
6. 2.5 * ATR STOPLOSS: Trailing stop for risk management

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d and 1w via mtf_data helper (call ONCE before loop each)
Position sizing: 0.25-0.35 discrete based on regime
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_dual_htf_hma_chop_regime_asymmetric_atr_v1"
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
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into a Gaussian normal distribution.
    Extreme values indicate potential reversals.
    """
    n = len(close)
    fisher = np.full(n, np.nan)
    trigger = np.full(n, np.nan)
    
    # Use median price (HL2)
    price = (high + low) / 2
    
    for i in range(period, n):
        # Calculate highest high and lowest low over period
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        
        # Avoid division by zero
        if hh - ll < 1e-10:
            fisher[i] = 0.0
            trigger[i] = fisher[i]
            continue
        
        # Calculate normalized price (0.66 weighting for smoothing)
        price_norm = 0.66 * ((price[i] - ll) / (hh - ll) - 0.5) + \
                     0.67 * (0.66 * ((price[i-1] - ll) / (hh - ll) - 0.5) if i > period else 0)
        
        # Clamp to avoid extreme values (prevent log errors)
        price_norm = np.clip(price_norm, -0.999, 0.999)
        
        # Fisher transform: 0.5 * ln((1+x)/(1-x))
        fisher[i] = 0.5 * np.log((1 + price_norm) / (1 - price_norm))
        
        # Smooth fisher value
        if i > period:
            fisher[i] = 0.67 * fisher[i] + 0.33 * fisher[i-1]
        
        # Trigger line (previous fisher value)
        trigger[i] = fisher[i-1] if i > period else fisher[i]
    
    return fisher, trigger

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        # Highest high and lowest low over period
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        
        if hh - ll < 1e-10:
            continue
        
        # Sum of ATR over period (True Range)
        tr_sum = 0
        for j in range(i-period+1, i+1):
            tr1 = high[j] - low[j]
            tr2 = np.abs(high[j] - close[j-1]) if j > 0 else tr1
            tr3 = np.abs(low[j] - close[j-1]) if j > 0 else tr1
            tr_sum += np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Choppiness calculation
        chop[i] = 100 * np.log10(tr_sum / (hh - ll)) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    fisher, fisher_trigger = calculate_fisher_transform(high, low, close, 9)
    chop = calculate_choppiness_index(high, low, close, 14)
    
    # Calculate EMA21 for trend pullback entries
    ema_21 = pd.Series(close).ewm(span=21, min_periods=21, adjust=False).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels based on regime (Rule 4)
    SIZE_TREND = 0.35
    SIZE_RANGE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # === DUAL HTF HMA TREND BIAS ===
        # Both 1d and 1w must agree for strong bias
        bull_bias_strong = close[i] > hma_1d_aligned[i] and close[i] > hma_1w_aligned[i]
        bear_bias_strong = close[i] < hma_1d_aligned[i] and close[i] < hma_1w_aligned[i]
        bull_bias_weak = close[i] > hma_1d_aligned[i]
        bear_bias_weak = close[i] < hma_1d_aligned[i]
        
        # === CHOPPINESS REGIME ===
        is_ranging = chop[i] > 61.8
        is_trending = chop[i] < 38.2
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        current_size = SIZE_RANGE  # Default to range sizing
        
        # RANGING MARKET: Use Fisher Transform reversals
        if is_ranging:
            current_size = SIZE_RANGE
            # Long: Fisher crosses above -1.5 from below + bullish bias
            if fisher[i] > -1.5 and fisher_trigger[i] <= -1.5 and bull_bias_weak:
                new_signal = current_size
            # Short: Fisher crosses below +1.5 from above + bearish bias
            elif fisher[i] < 1.5 and fisher_trigger[i] >= 1.5 and bear_bias_weak:
                new_signal = -current_size
        
        # TRENDING MARKET: Use pullback entries
        elif is_trending:
            current_size = SIZE_TREND
            if bull_bias_strong:
                # Long pullback: price near EMA21 in uptrend
                if close[i] < ema_21[i] * 1.02 and close[i] > ema_21[i] * 0.98:
                    new_signal = current_size
            elif bear_bias_strong:
                # Short pullback: price near EMA21 in downtrend
                if close[i] > ema_21[i] * 0.98 and close[i] < ema_21[i] * 1.02:
                    new_signal = -current_size
        
        # NEUTRAL: Conservative Fisher entries with strong bias only
        else:
            current_size = SIZE_RANGE
            if fisher[i] < -1.8 and bull_bias_strong:
                new_signal = current_size
            elif fisher[i] > 1.8 and bear_bias_strong:
                new_signal = -current_size
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === REGIME CHANGE EXIT ===
        # Exit if regime changes against position type
        if in_position and new_signal != 0.0:
            if position_side > 0 and is_trending and not bull_bias_strong:
                new_signal = 0.0
            if position_side < 0 and is_trending and not bear_bias_strong:
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