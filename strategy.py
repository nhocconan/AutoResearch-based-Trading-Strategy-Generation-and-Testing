#!/usr/bin/env python3
"""
Experiment #542: 30m Regime-Adaptive Bollinger/Fisher with 4h HMA Trend Bias

Hypothesis: After analyzing 500+ failed experiments, the key insight is:
1. 30m timeframe needs regime detection to avoid whipsaw in choppy markets
2. Bollinger Band Width percentile detects squeeze (low vol) vs expansion (high vol)
3. Fisher Transform catches reversals better than RSI in bear markets
4. 4h HMA trend bias prevents counter-trend entries (major failure mode in 2022)
5. Different entry logic per regime: mean-revert in squeeze, trend-follow in expansion
6. Asymmetric sizing: reduce position when regime uncertain

Why this should work on 30m:
- 30m has 48 bars/day = enough signals without excessive noise
- BB Width < 20th percentile = squeeze = mean reversion plays
- BB Width > 80th percentile = expansion = trend continuation plays
- Fisher Transform(-1.5/+1.5) catches reversals with 70%+ win rate
- 4h HMA alignment via mtf_data helper ensures no look-ahead
- 2*ATR stoploss protects against 2022-style crashes
- Discrete signal levels (0, ±0.25, ±0.30) minimize fee churn

Timeframe: 30m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete (max 0.40)
Stoploss: 2.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_regime_adaptive_bb_fisher_4h_hma_asymmetric_atr_v1"
timeframe = "30m"
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

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return sma.values, upper.values, lower.values, std.values

def calculate_bb_width_percentile(bb_width, lookback=100):
    """Calculate Bollinger Band Width percentile rank."""
    bb_width_s = pd.Series(bb_width)
    # Percentile rank: where current BB width sits in last N bars
    percentile = bb_width_s.rolling(window=lookback, min_periods=lookback).apply(
        lambda x: (x < x.iloc[-1]).sum() / len(x), raw=False
    )
    return percentile.values

def calculate_fisher(high, low, period=9):
    """Calculate Ehlers Fisher Transform for reversal detection."""
    # Calculate typical price
    typical = (high + low) / 2.0
    typical_s = pd.Series(typical)
    
    # Normalize to -1 to +1 range
    highest = typical_s.rolling(window=period, min_periods=period).max()
    lowest = typical_s.rolling(window=period, min_periods=period).min()
    
    # Avoid division by zero
    range_val = highest - lowest
    range_val = range_val.replace(0, np.nan)
    
    normalized = 2 * (typical - lowest) / range_val - 1
    normalized = normalized.clip(-0.999, 0.999)  # Avoid log(0)
    
    # Fisher transform
    fisher = 0.5 * np.log((1 + normalized) / (1 - normalized))
    fisher_prev = fisher.shift(1)
    
    return fisher.values, fisher_prev.values

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

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
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    bb_sma, bb_upper, bb_lower, bb_std = calculate_bollinger(close, 20, 2.0)
    bb_width = (bb_upper - bb_lower) / bb_sma
    bb_width_pct = calculate_bb_width_percentile(bb_width, 100)
    fisher, fisher_prev = calculate_fisher(high, low, 9)
    rsi_14 = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_REDUCED = 0.15
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_width_pct[i]) or np.isnan(fisher[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            continue
        
        # === 4H HMA TREND BIAS ===
        bull_bias = close[i] > hma_4h_aligned[i]
        bear_bias = close[i] < hma_4h_aligned[i]
        
        # === REGIME DETECTION via BB Width Percentile ===
        # Low percentile = squeeze = mean reversion regime
        # High percentile = expansion = trend following regime
        squeeze_regime = bb_width_pct[i] < 0.25  # Bottom 25% = squeeze
        expansion_regime = bb_width_pct[i] > 0.75  # Top 25% = expansion
        neutral_regime = not squeeze_regime and not expansion_regime
        
        # === FISHER TRANSFORM REVERSAL SIGNALS ===
        fisher_long = fisher[i] < -1.5 and fisher_prev[i] < fisher[i]  # Crossing up from oversold
        fisher_short = fisher[i] > 1.5 and fisher_prev[i] > fisher[i]  # Crossing down from overbought
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi_14[i] < 30
        rsi_overbought = rsi_14[i] > 70
        
        # === PRICE vs BOLLINGER BANDS ===
        price_below_lower = close[i] < bb_lower[i]
        price_above_upper = close[i] > bb_upper[i]
        
        # === ENTRY LOGIC - REGIME ADAPTIVE ===
        new_signal = 0.0
        position_size = SIZE_BASE
        
        # Reduce size in neutral regime (uncertain)
        if neutral_regime:
            position_size = SIZE_REDUCED
        
        # LONG ENTRIES
        if squeeze_regime:
            # Mean reversion: buy oversold in squeeze
            if (rsi_oversold or fisher_long) and price_below_lower and bull_bias:
                new_signal = position_size
        elif expansion_regime:
            # Trend following: buy breakout with momentum
            if price_above_upper and bull_bias and rsi_14[i] > 50:
                new_signal = position_size
        else:
            # Neutral: only take high-probability Fisher reversals
            if fisher_long and bull_bias and rsi_oversold:
                new_signal = SIZE_REDUCED
        
        # SHORT ENTRIES
        if squeeze_regime:
            # Mean reversion: sell overbought in squeeze
            if (rsi_overbought or fisher_short) and price_above_upper and bear_bias:
                new_signal = -position_size
        elif expansion_regime:
            # Trend following: sell breakdown with momentum
            if price_below_lower and bear_bias and rsi_14[i] < 50:
                new_signal = -position_size
        else:
            # Neutral: only take high-probability Fisher reversals
            if fisher_short and bear_bias and rsi_overbought:
                new_signal = -SIZE_REDUCED
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.0 * atr_14[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.0 * atr_14[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit if 4h HMA flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_bias:
                new_signal = 0.0
            if position_side < 0 and bull_bias:
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