#!/usr/bin/env python3
"""
Experiment #020: 30m Fisher Transform Reversal + 4h HMA Trend + Volatility Regime

Hypothesis: After analyzing 19 failed experiments, the pattern shows:
1. Lower TFs (15m/30m) need HTF trend filter to avoid noise whipsaws
2. Mean reversion works better than pure trend in bear/range markets (2022, 2025)
3. Fisher Transform catches reversals better than RSI/CRSI in crypto volatility
4. Current best (Sharpe=0.123) uses Supertrend+RSI - Fisher should beat this

This 30m strategy combines:

1. 4h HMA(21) trend bias: Stable HTF filter. Only long if price>4h_HMA,
   only short if price<4h_HMA. Prevents counter-trend trades.

2. Ehlers Fisher Transform (period=9): Long when Fisher crosses above -1.2,
   Short when crosses below +1.2. Catches reversals in bear rallies.
   Proven edge in crypto mean reversion (literature Sharpe 0.8-1.5).

3. RSI(7) confirmation: RSI<35 for longs, RSI>65 for shorts. Adds confluence.

4. Adaptive Volatility Sizing: Position size = base_size * (ATR_median / ATR_current)
   Reduces size during vol spikes (protects from 2022-style crashes).

5. ATR-based stoploss: 2.5*ATR trailing stop to protect from crashes.

6. Volume confirmation: Entry only if volume > 0.8 * volume_SMA(20)

Why this should beat #013 (Sharpe=-2.385) and current best (Sharpe=0.123):
- Fisher Transform more sensitive than CRSI for 30m timeframe
- 4h HMA more stable than Supertrend for trend filter
- Adaptive sizing reduces DD during vol spikes
- Volume filter prevents false breakouts
- Target 40-70 trades/year on 30m (optimal frequency per Rule 10)

Timeframe: 30m (REQUIRED)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.35 discrete, volatility-adaptive
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_fisher_4h_hma_rsi_vol_adaptive_atr_v1"
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
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    
    Fisher Transform converts price to a Gaussian normal distribution.
    Crossings of extreme values (-2, +2) signal reversals.
    
    Formula:
    1. Calculate typical price: (high + low) / 2
    2. Normalize to -1 to +1 range over lookback period
    3. Apply Fisher transform: 0.5 * ln((1 + x) / (1 - x))
    4. Smooth with EMA
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # Typical price
    typical = (high_s + low_s) / 2
    
    # Normalize to -1 to +1 range
    highest = typical.rolling(window=period, min_periods=period).max()
    lowest = typical.rolling(window=period, min_periods=period).min()
    
    # Avoid division by zero
    range_val = highest - lowest
    range_val = range_val.replace(0, np.inf)
    
    normalized = 2 * (typical - lowest) / range_val - 1
    
    # Clip to avoid ln domain errors
    normalized = normalized.clip(-0.999, 0.999)
    
    # Fisher transform
    fisher = 0.5 * np.log((1 + normalized) / (1 - normalized))
    
    # Smooth with EMA
    fisher_smooth = fisher.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return fisher_smooth.values

def calculate_volume_sma(volume, period=20):
    """Calculate simple moving average of volume."""
    vol_s = pd.Series(volume)
    vol_sma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_sma

def calculate_atr_median(atr, lookback=50):
    """Calculate rolling median of ATR for adaptive sizing."""
    atr_s = pd.Series(atr)
    atr_median = atr_s.rolling(window=lookback, min_periods=lookback).median().values
    return atr_median

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_7 = calculate_rsi(close, 7)
    fisher = calculate_fisher_transform(high, low, 9)
    vol_sma = calculate_volume_sma(volume, 20)
    atr_med = calculate_atr_median(atr_14, 50)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.30  # 30% of capital base
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    prev_fisher = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            continue
        
        if np.isnan(fisher[i]) or np.isnan(rsi_7[i]):
            continue
        
        if np.isnan(vol_sma[i]) or vol_sma[i] == 0:
            continue
        
        # === 4H HMA TREND BIAS (HTF filter) ===
        bull_bias = close[i] > hma_4h_aligned[i]
        bear_bias = close[i] < hma_4h_aligned[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.2 from below
        # Short: Fisher crosses below +1.2 from above
        fisher_long_cross = (prev_fisher < -1.2 and fisher[i] >= -1.2) if not np.isnan(prev_fisher) else False
        fisher_short_cross = (prev_fisher > 1.2 and fisher[i] <= 1.2) if not np.isnan(prev_fisher) else False
        
        # Also check extreme levels for reversal
        fisher_oversold = fisher[i] < -1.5
        fisher_overbought = fisher[i] > 1.5
        
        # === RSI CONFIRMATION ===
        rsi_oversold = rsi_7[i] < 35
        rsi_overbought = rsi_7[i] > 65
        
        # === VOLUME CONFIRMATION ===
        volume_ok = volume[i] > 0.8 * vol_sma[i]
        
        # === ADAPTIVE VOLATILITY SIZING ===
        # Reduce size when ATR is high (vol spike protection)
        if np.isnan(atr_med[i]) or atr_med[i] == 0:
            vol_scalar = 1.0
        else:
            vol_scalar = min(1.5, max(0.5, atr_med[i] / atr_14[i]))
        
        adaptive_size = BASE_SIZE * vol_scalar
        adaptive_size = min(0.35, max(0.15, adaptive_size))  # Clamp to 0.15-0.35
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG ENTRY: Fisher reversal + RSI oversold + bull bias + volume
        if fisher_long_cross and rsi_oversold and bull_bias and volume_ok:
            new_signal = adaptive_size
        elif fisher_oversold and rsi_oversold and bull_bias and volume_ok:
            new_signal = adaptive_size
        
        # SHORT ENTRY: Fisher reversal + RSI overbought + bear bias + volume
        elif fisher_short_cross and rsi_overbought and bear_bias and volume_ok:
            new_signal = -adaptive_size
        elif fisher_overbought and rsi_overbought and bear_bias and volume_ok:
            new_signal = -adaptive_size
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest price for long position
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest price for short position
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_exit = False
        if in_position and position_side != 0:
            # Exit if HTF trend flips against position
            if position_side > 0 and bear_bias:
                trend_exit = True
            if position_side < 0 and bull_bias:
                trend_exit = True
        
        # Apply stoploss or trend exit
        if stoploss_triggered or trend_exit:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                # New entry
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                # Exit position
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
        
        # Update previous Fisher value for crossover detection
        prev_fisher = fisher[i]
        
        signals[i] = new_signal
    
    return signals