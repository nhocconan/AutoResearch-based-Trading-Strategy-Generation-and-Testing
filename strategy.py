#!/usr/bin/env python3
"""
Experiment #005: 1h Fisher-Choppiness Regime with 4h HMA Bias

Hypothesis: After 4 consecutive failures with over-filtered regime strategies,
the issue is TOO MANY conflicting conditions. This strategy simplifies:

1. 4H HMA(21) for trend bias (proven stable filter)
2. 1h Fisher Transform(9) for entry timing (better reversal detection than RSI)
3. Choppiness Index(14) for regime: <40=trend, >60=range
4. Relaxed volume filter (0.7x avg, not 0.8x)
5. Session filter 6-22 UTC (wider than 8-20)
6. Looser Fisher thresholds (-1.8/+1.8 extremes, -1.5/+1.5 crosses)
7. 2.5 ATR stoploss (wider to avoid premature exits)
8. Position size 0.20-0.30 discrete

Key changes from failed strategies:
- FEWER filters = more trades (target 40-80/year)
- Looser thresholds ensure entries happen in all market conditions
- ATR-based size scaling protects in high vol (2022 crash)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_chop_4h_hma_session_v1"
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

def calculate_fisher(close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Fisher = 0.5 * ln((1 + X) / (1 - X)) where X = normalized price
    Excellent for catching reversals at extremes (-2.0 to +2.0)
    """
    close_s = pd.Series(close)
    
    highest = close_s.rolling(window=period, min_periods=period).max()
    lowest = close_s.rolling(window=period, min_periods=period).min()
    
    price_range = highest - lowest
    price_range = price_range.replace(0, 0.001)
    
    x = ((close_s - lowest) / price_range * 2 - 1).clip(-0.99, 0.99)
    fisher = 0.5 * np.log((1 + x) / (1 - x))
    fisher_signal = fisher.shift(1)
    
    return fisher.values, fisher_signal.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 60 = range/choppy, CHOP < 40 = trending
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr_sum = tr.rolling(window=period, min_periods=period).sum()
    highest_high = high_s.rolling(window=period, min_periods=period).max()
    lowest_low = low_s.rolling(window=period, min_periods=period).min()
    
    price_range = highest_high - lowest_low
    chop = 100 * np.log10(atr_sum / price_range.replace(0, np.inf)) / np.log10(period)
    
    return chop.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    fisher, fisher_signal = calculate_fisher(close, 9)
    chop_14 = calculate_choppiness(high, low, close, 14)
    
    # Volume SMA for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR median for scaling (computed incrementally in loop)
    atr_median = np.nanmedian(atr_14[100:200]) if n > 200 else np.nanmedian(atr_14[100:])
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            continue
        
        if np.isnan(chop_14[i]):
            continue
        
        if np.isnan(vol_sma[i]) or vol_sma[i] == 0:
            continue
        
        # === SESSION FILTER (6-22 UTC for liquidity) ===
        hour = pd.to_datetime(open_time[i], unit='ms').hour
        in_session = 6 <= hour <= 22
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_trend_regime = chop_14[i] < 40
        is_range_regime = chop_14[i] > 60
        
        # === 4H HMA TREND BIAS ===
        bull_bias = close[i] > hma_4h_aligned[i]
        bear_bias = close[i] < hma_4h_aligned[i]
        
        # === VOLUME CONFIRMATION (relaxed: 0.7x avg) ===
        volume_confirmed = volume[i] > 0.7 * vol_sma[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        # Fisher crossing above -1.5 from below (bullish reversal)
        fisher_long_cross = (fisher[i] > -1.5) and (fisher_signal[i] <= -1.5)
        # Fisher crossing below +1.5 from above (bearish reversal)
        fisher_short_cross = (fisher[i] < 1.5) and (fisher_signal[i] >= 1.5)
        
        # Fisher at extreme (for range regime mean reversion)
        fisher_oversold = fisher[i] < -1.8
        fisher_overbought = fisher[i] > 1.8
        
        # === ATR-BASED POSITION SIZING ===
        if i > 150:
            atr_median = np.nanmedian(atr_14[100:i])
        atr_ratio = atr_14[i] / atr_median if atr_median > 0 else 1.0
        atr_ratio = np.clip(atr_ratio, 0.6, 1.8)
        size_multiplier = 1.0 / atr_ratio
        current_size = BASE_SIZE * size_multiplier
        current_size = np.clip(current_size, 0.18, 0.32)
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # MODE 1: TREND REGIME - Fisher crossover with HTF bias
        if is_trend_regime:
            # Long: Fisher cross + 4h bullish + volume
            if fisher_long_cross and bull_bias and volume_confirmed and in_session:
                new_signal = current_size
            # Short: Fisher cross + 4h bearish + volume
            elif fisher_short_cross and bear_bias and volume_confirmed and in_session:
                new_signal = -current_size
        
        # MODE 2: RANGE REGIME - Fisher extreme mean reversion
        elif is_range_regime:
            # Long: Fisher oversold (allow if not strongly bearish)
            if fisher_oversold and in_session:
                if not bear_bias or chop_14[i] > 55:
                    new_signal = current_size
            # Short: Fisher overbought (allow if not strongly bullish)
            elif fisher_overbought and in_session:
                if not bull_bias or chop_14[i] > 55:
                    new_signal = -current_size
        
        # MODE 3: TRANSITION REGIME - Either crossover or extreme
        else:
            # Long: Fisher cross or oversold + not bearish
            if (fisher_long_cross or fisher_oversold) and not bear_bias and in_session:
                new_signal = current_size
            # Short: Fisher cross or overbought + not bullish
            elif (fisher_short_cross or fisher_overbought) and not bull_bias and in_session:
                new_signal = -current_size
        
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
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if 4h bias turns strongly bearish in trend regime
            if position_side > 0 and bear_bias and is_trend_regime:
                trend_reversal = True
            # Exit short if 4h bias turns strongly bullish in trend regime
            if position_side < 0 and bull_bias and is_trend_regime:
                trend_reversal = True
        
        # Apply stoploss or trend reversal
        if stoploss_triggered or trend_reversal:
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
        
        signals[i] = new_signal
    
    return signals