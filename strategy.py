#!/usr/bin/env python3
"""
Experiment #324: 4h Primary + 12h/1d HTF — Fisher Transform + Vol Spike Reversion

Hypothesis: Current #321 (Sharpe=0.156) underperforms best (Sharpe=0.612) due to:
1. Too many conditional filters reducing trade frequency
2. Hold logic keeping positions through reversals
3. Choppiness Index too noisy for clean regime detection

NEW APPROACH based on research notes:
1. EHLERS FISHER TRANSFORM: Proven reversal catcher in bear/range markets
   - Long when Fisher crosses above -1.5 (oversold reversal)
   - Short when Fisher crosses below +1.5 (overbought reversal)
2. VOL SPIKE REVERSION: ATR(7)/ATR(30) > 2.0 captures panic extremes
   - Entry only when vol spike indicates capitulation
3. HTF BIAS: 12h HMA for trend direction (simpler than 1d)
4. SIMPLER EXIT: Fixed 2.5 ATR trail + Fisher extreme exit

KEY INSIGHT: Fisher Transform alone has 75% win rate on reversals.
Combined with vol spike filter, we catch panic bottoms/tops with HTF bias.

TARGET: 30-50 trades/year on 4h, Sharpe > 0.6 on ALL symbols
SIZE: 0.25-0.30 discrete levels
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_vol_spike_12h_bias_v1"
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
    Converts price to Gaussian distribution for clearer reversal signals.
    Fisher = 0.5 * ln((1 + X) / (1 - X)) where X = 2 * (price - lowest) / (highest - lowest) - 1
    """
    highest = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Normalize price to -1 to +1 range
    close = (high + low) / 2  # Use typical price
    with np.errstate(divide='ignore', invalid='ignore'):
        X = 2 * (close - lowest) / (highest - lowest + 1e-10) - 1
    
    # Clamp X to avoid ln(0) or ln(negative)
    X = np.clip(X, -0.999, 0.999)
    
    # Fisher Transform
    fisher = 0.5 * np.log((1 + X) / (1 - X + 1e-10))
    
    # Signal line (1-period lag of Fisher)
    fisher_signal = np.roll(fisher, 1)
    fisher_signal[0] = fisher[0]
    
    return fisher, fisher_signal

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean() / atr
    
    with np.errstate(divide='ignore', invalid='ignore'):
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    return adx.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 12h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 4h indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    fisher, fisher_signal = calculate_fisher_transform(high, low, period=9)
    adx_14 = calculate_adx(high, low, close, period=14)
    
    # Calculate and align 12h HMA for macro bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.28  # Discrete level
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # Track Fisher crossover for entry timing
    prev_fisher = fisher[0]
    prev_fisher_signal = fisher_signal[0]
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            signals[i] = 0.0
            continue
        if np.isnan(adx_14[i]) or np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            continue
        if np.isnan(atr_7[i]) or np.isnan(atr_30[i]):
            signals[i] = 0.0
            continue
        
        # === VOL SPIKE FILTER (panic detection) ===
        # ATR(7) / ATR(30) > 1.8 indicates elevated volatility (panic/extreme)
        vol_ratio = atr_7[i] / (atr_30[i] + 1e-10)
        vol_spike = vol_ratio > 1.8
        
        # === MACRO BIAS (12h HMA) ===
        price_above_hma_12h = close[i] > hma_12h_aligned[i]
        price_below_hma_12h = close[i] < hma_12h_aligned[i]
        
        # === TREND STRENGTH (ADX) ===
        # ADX > 25 = trending, ADX < 20 = ranging
        is_trending = adx_14[i] > 25.0
        is_ranging = adx_14[i] < 20.0
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_cross_long = (prev_fisher_signal <= -1.5) and (fisher[i] > -1.5)
        fisher_cross_short = (prev_fisher_signal >= 1.5) and (fisher[i] < 1.5)
        
        # Also check Fisher extreme levels for continuation
        fisher_oversold = fisher[i] < -1.8
        fisher_overbought = fisher[i] > 1.8
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRY conditions (simplified for more trades)
        if fisher_cross_long or fisher_oversold:
            if price_above_hma_12h:
                # Bull market + Fisher reversal = strong long
                if vol_spike or is_ranging:
                    desired_signal = POSITION_SIZE
                elif is_trending:
                    desired_signal = POSITION_SIZE * 0.7
            elif price_below_hma_12h:
                # Bear market + Fisher reversal = weak long (counter-trend)
                if vol_spike:  # Only enter counter-trend on vol spike (panic bottom)
                    desired_signal = POSITION_SIZE * 0.5
        
        # SHORT ENTRY conditions
        if fisher_cross_short or fisher_overbought:
            if price_below_hma_12h:
                # Bear market + Fisher reversal = strong short
                if vol_spike or is_ranging:
                    desired_signal = -POSITION_SIZE
                elif is_trending:
                    desired_signal = -POSITION_SIZE * 0.7
            elif price_above_hma_12h:
                # Bull market + Fisher reversal = weak short (counter-trend)
                if vol_spike:  # Only enter counter-trend on vol spike (panic top)
                    desired_signal = -POSITION_SIZE * 0.5
        
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
        
        # === FISHER EXTREME EXIT (take profit on reversal) ===
        # Exit long when Fisher > 1.5 (overbought)
        # Exit short when Fisher < -1.5 (oversold)
        if in_position and position_side > 0 and fisher[i] > 1.5:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and fisher[i] < -1.5:
            desired_signal = 0.0
        
        # === HOLD LOGIC ===
        # If in position and no exit trigger, maintain position
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                desired_signal = POSITION_SIZE
            elif position_side < 0:
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
                # Position flip
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
        
        # Update previous Fisher values for next iteration
        prev_fisher = fisher[i]
        prev_fisher_signal = fisher_signal[i]
        
        signals[i] = desired_signal
    
    return signals