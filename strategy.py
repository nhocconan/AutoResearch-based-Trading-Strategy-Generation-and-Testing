#!/usr/bin/env python3
"""
Experiment #328: 4h Fisher Transform Reversal with Dual HTF HMA Bias

Hypothesis: After analyzing failed strategies, Fisher Transform excels at catching 
reversals in bear/range markets (critical for 2025 test period). Unlike RSI which 
failed in mean-reversion configs (#318, #325), Fisher Transform normalizes price 
distribution and catches extremes better. Combined with 1d HMA for directional bias 
and 1w HMA for meta-trend confirmation, this should outperform pure trend following.

Key innovations:
1. Fisher Transform (period=9) - catches reversals at -1.5/+1.5 levels
2. Volume confirmation - taker_buy_volume ratio filters false breakouts
3. Dual HTF bias - 1d HMA required, 1w HMA boosts conviction
4. ATR trailing stop - 2.5x for proper risk management
5. Asymmetric sizing - larger positions when both HTF align

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d and 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_reversal_dual_htf_hma_volume_atr_v1"
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

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Normalizes price to Gaussian distribution, excellent for catching reversals.
    Reference: Ehlers, J.F. (2002) "Fishing Turnings with the Fisher Transform"
    """
    n = len(high)
    fisher = np.zeros(n)
    trigger = np.zeros(n)
    
    for i in range(period, n):
        # Calculate median price
        median = (high[i] + low[i]) / 2.0
        
        # Find highest high and lowest low over period
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        # Normalize to -1 to +1 range
        range_val = highest - lowest
        if range_val == 0:
            range_val = 0.001
        
        normalized = 0.999 * (median - lowest) / range_val + 0.001
        
        # Fisher calculation
        if normalized >= 1.0:
            normalized = 0.999
        if normalized <= 0.0:
            normalized = 0.001
            
        fisher_val = 0.5 * np.log((1 + normalized) / (1 - normalized))
        
        # Smooth with previous value
        if i > period:
            fisher[i] = 0.6 * fisher_val + 0.4 * fisher[i-1]
            trigger[i] = fisher[i-1]
        else:
            fisher[i] = fisher_val
            trigger[i] = fisher_val
    
    return fisher, trigger

def calculate_volume_ratio(taker_buy_volume, volume):
    """Calculate taker buy volume ratio for sentiment."""
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(volume > 0, taker_buy_volume / volume, 0.5)
    ratio = np.nan_to_num(ratio, nan=0.5, posinf=0.5, neginf=0.5)
    return ratio

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
    
    rsi = rsi.replace([np.inf, -np.inf], np.nan)
    rsi = rsi.fillna(50.0)
    return rsi.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_vol = prices["taker_buy_volume"].values
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
    atr = calculate_atr(high, low, close, 14)
    fisher, fisher_trigger = calculate_fisher_transform(high, low, 9)
    volume_ratio = calculate_volume_ratio(taker_buy_vol, volume)
    rsi = calculate_rsi(close, 14)
    
    # Calculate volume ratio SMA for confirmation
    vol_ratio_sma = pd.Series(volume_ratio).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    SIZE_MAX = 0.35
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(fisher[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(volume_ratio[i]) or np.isnan(vol_ratio_sma[i]):
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME BIAS ===
        # 1d HMA = primary directional bias (REQUIRED for entry direction)
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # 1w HMA = meta-trend confirmation (boosts position size)
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 from below (oversold reversal)
        fisher_long = (fisher[i] > -1.5) and (fisher_trigger[i] <= -1.5)
        
        # Short: Fisher crosses below +1.5 from above (overbought reversal)
        fisher_short = (fisher[i] < 1.5) and (fisher_trigger[i] >= 1.5)
        
        # Alternative: Fisher extreme levels (for additional entries)
        fisher_extreme_long = fisher[i] < -2.0
        fisher_extreme_short = fisher[i] > 2.0
        
        # === VOLUME CONFIRMATION ===
        # Bullish volume: taker buy ratio > 0.55 (buying pressure)
        vol_bullish = volume_ratio[i] > 0.55
        
        # Bearish volume: taker buy ratio < 0.45 (selling pressure)
        vol_bearish = volume_ratio[i] < 0.45
        
        # Volume spike confirmation
        vol_spike = volume_ratio[i] > vol_ratio_sma[i] * 1.2 if not np.isnan(vol_ratio_sma[i]) else False
        
        # === RSI FILTER (avoid entering against extreme momentum) ===
        rsi_neutral = (rsi[i] > 30) and (rsi[i] < 70)
        rsi_bullish = rsi[i] > 50
        rsi_bearish = rsi[i] < 50
        
        # === DETERMINE POSITION SIZE ===
        # Base size
        position_size = SIZE_BASE
        
        # Boost size if both HTF align
        if bull_trend_1d and bull_trend_1w:
            position_size = SIZE_STRONG
        elif bear_trend_1d and bear_trend_1w:
            position_size = SIZE_STRONG
        
        # Max size with volume confirmation
        if position_size == SIZE_STRONG and vol_spike:
            position_size = SIZE_MAX
        
        # === ENTRY CONDITIONS ===
        new_signal = 0.0
        
        # LONG: 1d bias up + Fisher reversal + volume confirmation
        # Looser conditions for trade generation (Rule 9)
        long_conditions = (
            bull_trend_1d and
            (fisher_long or fisher_extreme_long) and
            (vol_bullish or rsi_bullish)  # Either volume OR RSI confirmation
        )
        
        # SHORT: 1d bias down + Fisher reversal + volume confirmation
        short_conditions = (
            bear_trend_1d and
            (fisher_short or fisher_extreme_short) and
            (vol_bearish or rsi_bearish)  # Either volume OR RSI confirmation
        )
        
        # === GENERATE SIGNAL ===
        if long_conditions:
            new_signal = position_size
        
        if short_conditions:
            new_signal = -position_size
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long positions
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short positions
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit long if 1d trend turns bearish
        if in_position and position_side > 0 and bear_trend_1d:
            new_signal = 0.0
        
        # Exit short if 1d trend turns bullish
        if in_position and position_side < 0 and bull_trend_1d:
            new_signal = 0.0
        
        # === FISHER REVERSAL EXIT ===
        # Exit long if Fisher goes above +1.5 (overbought)
        if in_position and position_side > 0 and fisher[i] > 1.5:
            new_signal = 0.0
        
        # Exit short if Fisher goes below -1.5 (oversold)
        if in_position and position_side < 0 and fisher[i] < -1.5:
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