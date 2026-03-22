#!/usr/bin/env python3
"""
Experiment #025: 1h Regime-Adaptive with 4h/1d HMA Trend Confirmation

Hypothesis: Previous regime strategies failed due to overly strict entry conditions.
This version uses SIMPLER regime detection + LOOSER entry thresholds to ensure
trades are generated while maintaining quality through HTF confirmation.

Core Logic:
1. 4h HMA(21) determines PRIMARY trend direction (long bias if price > HMA)
2. 1d HMA(21) confirms MAJOR trend (avoid counter-trend trades)
3. Choppiness Index(14) detects regime: >55 = range, <45 = trend
4. 1h RSI(7) for entry timing (looser thresholds: 35/65 instead of 20/80)
5. Volume filter: only enter if volume > 0.7x 20-bar average
6. Session filter: prefer 8-20 UTC (higher liquidity)
7. ATR(14) trailing stop: 2.5x for capital protection

Key Changes from Failed Experiments:
- RSI thresholds relaxed (35/65 vs 20/80) to generate MORE trades
- Choppiness simplified (binary regime, not complex scoring)
- Reduced confluence requirement (2+ filters vs 3+)
- Position size: 0.25 discrete (smaller for 1h TF to reduce fee impact)
- Target: 40-80 trades/year (strict enough to avoid fee drag)

Timeframe: 1h (REQUIRED)
HTF: 4h and 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_adaptive_4h_1d_hma_rsi_chop_atr_v1"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = range/choppy, CHOP < 38.2 = trending
    """
    atr_vals = calculate_atr(high, low, close, period)
    
    # Sum of ATR over period
    atr_sum = pd.Series(atr_vals).rolling(window=period, min_periods=period).sum().values
    
    # Highest high and lowest low over period
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Price range
    price_range = highest_high - lowest_low
    
    # Avoid division by zero
    price_range = np.where(price_range == 0, 1e-10, price_range)
    
    # Choppiness calculation
    chop = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    
    # Clamp to valid range
    chop = np.clip(chop, 0, 100)
    
    return chop

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs moving average."""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / np.where(vol_ma == 0, 1e-10, vol_ma)
    return vol_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    
    # Calculate 1d indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_7 = calculate_rsi(close, 7)  # Faster RSI for entry timing
    rsi_14 = calculate_rsi(close, 14)  # Standard RSI for regime
    chop_14 = calculate_choppiness(high, low, close, 14)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
    # Extract hour from timestamp for session filter
    # Assuming open_time is in milliseconds since epoch
    try:
        hours = (prices["open_time"].values // 3600000) % 24
    except:
        hours = np.zeros(n)  # Fallback if open_time not available
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25  # 25% of capital per position
    
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
        
        if np.isnan(hma_4h_21_aligned[i]):
            continue
        
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(rsi_7[i]) or np.isnan(rsi_14[i]):
            continue
        
        if np.isnan(chop_14[i]):
            continue
        
        # === 4H PRIMARY TREND ===
        trend_4h_bullish = close[i] > hma_4h_21_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_21_aligned[i]
        
        # === 1D MAJOR TREND CONFIRMATION ===
        trend_1d_bullish = close[i] > hma_1d_21_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        # CHOP > 55 = range/choppy (mean reversion mode)
        # CHOP < 45 = trending (trend follow mode)
        # 45-55 = neutral (use trend bias)
        is_range_regime = chop_14[i] > 55
        is_trend_regime = chop_14[i] < 45
        
        # === RSI ENTRY SIGNALS (LOOSE THRESHOLDS) ===
        # Long: RSI(7) < 40 in uptrend OR RSI(7) < 35 in any regime
        # Short: RSI(7) > 60 in downtrend OR RSI(7) > 65 in any regime
        rsi_oversold = rsi_7[i] < 40
        rsi_overbought = rsi_7[i] > 60
        rsi_extreme_oversold = rsi_7[i] < 35
        rsi_extreme_overbought = rsi_7[i] > 65
        
        # === VOLUME FILTER ===
        # Only enter if volume >= 0.7x average (not too strict)
        volume_ok = vol_ratio[i] >= 0.7
        
        # === SESSION FILTER ===
        # Prefer 8-20 UTC (higher liquidity), but allow all hours with reduced size
        in_session = 8 <= hours[i] <= 20
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        if not in_session:
            current_size = BASE_SIZE * 0.6  # Reduced size outside session
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG ENTRY (2+ confluence required, not 3+ to ensure trades)
        long_score = 0
        
        # Trend alignment (4h bullish)
        if trend_4h_bullish:
            long_score += 1
        
        # Major trend confirmation (1d bullish) - optional but adds confidence
        if trend_1d_bullish:
            long_score += 0.5
        
        # RSI entry signal
        if rsi_oversold:
            long_score += 1
        if rsi_extreme_oversold:
            long_score += 0.5
        
        # Volume confirmation
        if volume_ok:
            long_score += 0.5
        
        # Session preference
        if in_session:
            long_score += 0.3
        
        # Range regime: prefer mean reversion (RSI extreme)
        if is_range_regime and rsi_extreme_oversold:
            long_score += 1
        
        # Trend regime: prefer trend pullback (RSI moderate + trend)
        if is_trend_regime and trend_4h_bullish and rsi_oversold:
            long_score += 1
        
        # Enter long if score >= 2.0 (relaxed from 3.0)
        if long_score >= 2.0:
            new_signal = current_size
        
        # SHORT ENTRY (2+ confluence required)
        short_score = 0
        
        # Trend alignment (4h bearish)
        if trend_4h_bearish:
            short_score += 1
        
        # Major trend confirmation (1d bearish)
        if trend_1d_bearish:
            short_score += 0.5
        
        # RSI entry signal
        if rsi_overbought:
            short_score += 1
        if rsi_extreme_overbought:
            short_score += 0.5
        
        # Volume confirmation
        if volume_ok:
            short_score += 0.5
        
        # Session preference
        if in_session:
            short_score += 0.3
        
        # Range regime: prefer mean reversion (RSI extreme)
        if is_range_regime and rsi_extreme_overbought:
            short_score += 1
        
        # Trend regime: prefer trend pullback (RSI moderate + trend)
        if is_trend_regime and trend_4h_bearish and rsi_overbought:
            short_score += 1
        
        # Enter short if score >= 2.0 (relaxed from 3.0)
        if short_score >= 2.0:
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
            # Exit long if 4h trend turns strongly bearish
            if position_side > 0 and trend_4h_bearish and rsi_7[i] > 55:
                trend_reversal = True
            # Exit short if 4h trend turns strongly bullish
            if position_side < 0 and trend_4h_bullish and rsi_7[i] < 45:
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