#!/usr/bin/env python3
"""
Experiment #102: 12h Primary + 1d/1w HTF — Fisher Transform + Donchian Breakout

Hypothesis: Previous regime-based strategies over-complicated entry logic, resulting in
too few trades or whipsaw losses. Fisher Transform is proven to catch reversals in
bear/range markets (like 2025), while Donchian breakout provides clean trend confirmation.
Combined with dual HTF filter (1d slope + 1w position), this should generate 25-40 trades/year
with better risk-adjusted returns.

Strategy Logic:
1. FISHER TRANSFORM (9): Long when Fisher crosses above -1.5, Short when crosses below +1.5
2. DONCHIAN CHANNEL (20): Price breaking 20-bar high/low confirms trend direction
3. 1d HMA(21) SLOPE: Intermediate trend bias (only long if slope > 0, short if < 0)
4. 1w HMA(50) POSITION: Ultra-long-term filter (price above = bullish bias, below = bearish)
5. ATR(14) stoploss: 2.5x trailing stop + position sizing adjustment
6. Position size: 0.25-0.35 discrete based on vol regime

Why this should work:
- Fisher Transform excels at reversal detection in bear/range markets
- Donchian breakout is simple, proven trend confirmation (no complex regime detection)
- Dual HTF filter (1d + 1w) prevents counter-trend trades without over-filtering
- 12h timeframe naturally limits trades to target range
- Simpler logic = more trades = better statistics across all symbols

Timeframe: 12h (REQUIRED)
HTF: 1d and 1w via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25-0.35 discrete
Stoploss: 2.5 * ATR(14) trailing
Target trades: 25-40/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_fisher_donchian_hma_1d1w_v1"
timeframe = "12h"
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

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into a Gaussian normal distribution for clearer reversal signals.
    Long when Fisher crosses above -1.5, Short when crosses below +1.5
    """
    # Calculate typical price
    typical = (high + low + close) / 3.0
    typical_s = pd.Series(typical)
    
    # Normalize price to -1 to +1 range
    highest = typical_s.rolling(window=period, min_periods=period).max().values
    lowest = typical_s.rolling(window=period, min_periods=period).min().values
    
    price_range = highest - lowest
    price_range = np.where(price_range == 0, 1e-10, price_range)
    
    normalized = (2.0 * (typical - lowest) / price_range) - 1.0
    normalized = np.clip(normalized, -0.999, 0.999)  # avoid log(0)
    
    # Fisher transform
    fisher = 0.5 * np.log((1 + normalized) / (1 - normalized))
    
    # Signal line (1-period lag of fisher)
    fisher_signal = np.roll(fisher, 1)
    fisher_signal[0] = fisher[0]
    
    return fisher, fisher_signal

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel upper and lower bands."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2.0
    return upper, lower, mid

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_hma_slope(hma_values, lookback=5):
    """Calculate HMA slope as percentage change over lookback period."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        if hma_values[i - lookback] != 0 and not np.isnan(hma_values[i - lookback]):
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback] * 100
    return slope

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_slope = calculate_hma_slope(hma_1d_21, 5)
    
    # Calculate 1w HTF indicators
    hma_1w_50 = calculate_hma(df_1w['close'].values, 50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope)
    hma_1w_50_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_50)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, 9)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, 20)
    
    # Additional trend confirmation
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.30
    HIGH_VOL_SIZE = 0.25  # Reduce size in high volatility
    LOW_VOL_SIZE = 0.35   # Increase size in low volatility
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_slope_aligned[i]):
            continue
        
        if np.isnan(hma_1w_50_aligned[i]):
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        if np.isnan(hma_21[i]) or np.isnan(hma_50[i]):
            continue
        
        # === 1W ULTRA-LONG TERM BIAS ===
        # Price above 1w HMA = bullish bias (prefer longs)
        # Price below 1w HMA = bearish bias (prefer shorts)
        price_above_1w_hma = close[i] > hma_1w_50_aligned[i]
        price_below_1w_hma = close[i] < hma_1w_50_aligned[i]
        
        # === 1D INTERMEDIATE TREND ===
        # HMA slope > 0.5 = bullish trend
        # HMA slope < -0.5 = bearish trend
        trend_1d_bullish = hma_1d_slope_aligned[i] > 0.5
        trend_1d_bearish = hma_1d_slope_aligned[i] < -0.5
        
        # Price vs 1d HMA
        price_above_1d_hma = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_21_aligned[i]
        
        # === 12H TREND CONFIRMATION ===
        hma_bullish = hma_21[i] > hma_50[i]
        hma_bearish = hma_21[i] < hma_50[i]
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i - 1] if i > 0 else False
        donchian_breakout_short = close[i] < donchian_lower[i - 1] if i > 0 else False
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_long_signal = (fisher[i] > -1.5) and (fisher_signal[i] <= -1.5)
        fisher_short_signal = (fisher[i] < 1.5) and (fisher_signal[i] >= 1.5)
        
        # Also allow Fisher extreme readings for mean reversion
        fisher_oversold = fisher[i] < -1.8
        fisher_overbought = fisher[i] > 1.8
        
        # === VOLATILITY REGIME FOR SIZING ===
        # Compare current ATR to 50-bar average ATR
        atr_avg_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
        if np.isnan(atr_avg_50[i]) or atr_avg_50[i] == 0:
            vol_ratio = 1.0
        else:
            vol_ratio = atr_14[i] / atr_avg_50[i]
        
        # Adjust position size based on volatility
        if vol_ratio > 1.5:
            current_size = HIGH_VOL_SIZE  # High vol = reduce size
        elif vol_ratio < 0.7:
            current_size = LOW_VOL_SIZE   # Low vol = increase size
        else:
            current_size = BASE_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Multiple confluence paths
        long_confidence = 0
        
        # Path 1: Fisher reversal + 1w bullish + 1d bullish
        if fisher_long_signal and price_above_1w_hma and (trend_1d_bullish or price_above_1d_hma):
            long_confidence = 3
        
        # Path 2: Fisher oversold + Donchian breakout + 1w bullish
        elif fisher_oversold and donchian_breakout_long and price_above_1w_hma:
            long_confidence = 3
        
        # Path 3: Donchian breakout + 1d bullish + 12h HMA bullish
        elif donchian_breakout_long and trend_1d_bullish and hma_bullish:
            long_confidence = 2
        
        # Path 4: Fisher oversold + 1w bullish (simpler, more trades)
        elif fisher_oversold and price_above_1w_hma and price_above_1d_hma:
            long_confidence = 2
        
        # Path 5: Frequency safeguard - allow weaker entry if no trades for 80 bars
        if bars_since_last_trade > 80 and long_confidence == 0 and not in_position:
            if fisher[i] < -1.0 and price_above_1w_hma:
                long_confidence = 1
        
        # SHORT ENTRIES - Multiple confluence paths
        short_confidence = 0
        
        # Path 1: Fisher reversal + 1w bearish + 1d bearish
        if fisher_short_signal and price_below_1w_hma and (trend_1d_bearish or price_below_1d_hma):
            short_confidence = 3
        
        # Path 2: Fisher overbought + Donchian breakout + 1w bearish
        elif fisher_overbought and donchian_breakout_short and price_below_1w_hma:
            short_confidence = 3
        
        # Path 3: Donchian breakout + 1d bearish + 12h HMA bearish
        elif donchian_breakout_short and trend_1d_bearish and hma_bearish:
            short_confidence = 2
        
        # Path 4: Fisher overbought + 1w bearish (simpler, more trades)
        elif fisher_overbought and price_below_1w_hma and price_below_1d_hma:
            short_confidence = 2
        
        # Path 5: Frequency safeguard - allow weaker entry if no trades for 80 bars
        if bars_since_last_trade > 80 and short_confidence == 0 and not in_position:
            if fisher[i] > 1.0 and price_below_1w_hma:
                short_confidence = 1
        
        # Set signal based on confidence
        if long_confidence >= 2:
            new_signal = current_size
        elif long_confidence == 1:
            new_signal = current_size * 0.6
        elif short_confidence >= 2:
            new_signal = -current_size
        elif short_confidence == 1:
            new_signal = -current_size * 0.6
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        # Exit if major trend reverses against position
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if 1w HMA flips bearish
            if position_side > 0 and price_below_1w_hma and trend_1d_bearish:
                trend_reversal = True
            # Exit short if 1w HMA flips bullish
            if position_side < 0 and price_above_1w_hma and trend_1d_bullish:
                trend_reversal = True
        
        # Apply stoploss or trend reversal
        if stoploss_triggered or trend_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals