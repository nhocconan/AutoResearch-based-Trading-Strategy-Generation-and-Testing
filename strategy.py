#!/usr/bin/env python3
"""
Experiment #209: 4h Primary + 1d HTF — Simplified Regime + RSI Mean Reversion

Hypothesis: Recent failures stem from overly complex multi-filter systems that rarely
all align. Research shows simpler mean reversion with HTF trend bias works better
on 4h timeframe. This strategy uses:

1. 1d HMA(21) SLOPE: Major trend direction (bullish/bearish bias)
2. 4h RSI(14): Entry timing at moderate extremes (30/70 not 20/80)
3. 4h Bollinger Bands(20, 2.0): Mean reversion zones
4. 4h ATR(14): Volatility filter + trailing stoploss
5. Regime simplification: Just trend vs range based on BB width percentile

Why this should work:
- Fewer filters = more trades (critical for meeting min trade requirements)
- 4h timeframe targets 20-50 trades/year (low fee drag)
- 1d HTF prevents fighting major trends
- Moderate RSI thresholds (30/70) trigger more often than extremes
- Asymmetric sizing: larger positions when HTF trend agrees

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
Target trades: 25-50/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_simp_rsi_bb_hma1d_v1"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bb_width = (upper - lower) / sma * 100
    return upper, lower, sma, bb_width

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_hma_slope(hma_values, lookback=5):
    """Calculate HMA slope as percentage change."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        if hma_values[i - lookback] != 0:
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback] * 100
    return slope

def calculate_bb_width_percentile(bb_width, period=100):
    """Calculate BB width percentile for regime detection."""
    bb_width_s = pd.Series(bb_width)
    percentile = bb_width_s.rolling(window=period, min_periods=period).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) * 100 if x.max() > x.min() else 50
    ).values
    percentile = np.nan_to_num(percentile, nan=50.0)
    return percentile

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_slope = calculate_hma_slope(hma_1d_21, 5)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_mid, bb_width = calculate_bollinger_bands(close, 20, 2.0)
    bb_width_pct = calculate_bb_width_percentile(bb_width, 100)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.28
    REDUCED_SIZE = 0.15
    
    # Track position state
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
        
        if np.isnan(rsi_14[i]) or np.isnan(bb_lower[i]):
            continue
        
        # === 1D TREND BIAS ===
        trend_1d_bullish = hma_1d_slope_aligned[i] > 0.5
        trend_1d_bearish = hma_1d_slope_aligned[i] < -0.5
        price_above_1d_hma = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_21_aligned[i]
        
        # === REGIME DETECTION ===
        is_low_vol = bb_width_pct[i] < 30  # Squeeze = potential breakout
        is_high_vol = bb_width_pct[i] > 70  # Extended = mean revert likely
        
        # === RSI SIGNALS (moderate thresholds for more trades) ===
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        rsi_extreme_low = rsi_14[i] < 25
        rsi_extreme_high = rsi_14[i] > 75
        
        # === BOLLINGER BAND POSITION ===
        price_below_bb_lower = close[i] < bb_lower[i]
        price_above_bb_upper = close[i] > bb_upper[i]
        price_near_bb_lower = close[i] < bb_lower[i] * 1.01
        price_near_bb_upper = close[i] > bb_upper[i] * 0.99
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        if not trend_1d_bullish and not trend_1d_bearish:
            current_size = REDUCED_SIZE  # Neutral trend = smaller size
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Multiple paths for trade frequency
        long_confidence = 0
        
        # Path 1: RSI oversold + BB lower (mean reversion)
        if rsi_oversold and price_below_bb_lower:
            long_confidence += 3
        
        # Path 2: RSI extreme + 1d bullish bias
        if rsi_extreme_low and trend_1d_bullish:
            long_confidence += 3
        
        # Path 3: Price near BB lower + 1d HMA support
        if price_near_bb_lower and price_above_1d_hma:
            long_confidence += 2
        
        # Path 4: Low vol squeeze + RSI oversold (breakout setup)
        if is_low_vol and rsi_oversold:
            long_confidence += 2
        
        # Path 5: Simple RSI oversold (fallback for trades)
        if rsi_14[i] < 30:
            long_confidence += 1
        
        # Size based on confidence
        if long_confidence >= 3:
            new_signal = current_size
        elif long_confidence >= 2:
            new_signal = current_size * 0.7
        elif long_confidence >= 1 and bars_since_last_trade > 60:
            new_signal = REDUCED_SIZE
        
        # SHORT ENTRIES
        short_confidence = 0
        
        # Path 1: RSI overbought + BB upper
        if rsi_overbought and price_above_bb_upper:
            short_confidence += 3
        
        # Path 2: RSI extreme + 1d bearish bias
        if rsi_extreme_high and trend_1d_bearish:
            short_confidence += 3
        
        # Path 3: Price near BB upper + 1d HMA resistance
        if price_near_bb_upper and price_below_1d_hma:
            short_confidence += 2
        
        # Path 4: Low vol squeeze + RSI overbought
        if is_low_vol and rsi_overbought:
            short_confidence += 2
        
        # Path 5: Simple RSI overbought
        if rsi_14[i] > 70:
            short_confidence += 1
        
        if short_confidence >= 3:
            new_signal = -current_size
        elif short_confidence >= 2:
            new_signal = -current_size * 0.7
        elif short_confidence >= 1 and bars_since_last_trade > 60:
            new_signal = -REDUCED_SIZE
        
        # === FREQUENCY SAFEGUARD ===
        # Force trade if no signal for 120 bars (~20 days on 4h)
        if bars_since_last_trade > 120 and new_signal == 0.0 and not in_position:
            if trend_1d_bullish and rsi_14[i] < 40:
                new_signal = REDUCED_SIZE
            elif trend_1d_bearish and rsi_14[i] > 60:
                new_signal = -REDUCED_SIZE
            elif rsi_14[i] < 28:
                new_signal = REDUCED_SIZE * 0.7
            elif rsi_14[i] > 72:
                new_signal = -REDUCED_SIZE * 0.7
        
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
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Exit long if 1d trend turns bearish strongly
            if position_side > 0 and trend_1d_bearish and hma_1d_slope_aligned[i] < -1.0:
                regime_reversal = True
            # Exit short if 1d trend turns bullish strongly
            if position_side < 0 and trend_1d_bullish and hma_1d_slope_aligned[i] > 1.0:
                regime_reversal = True
        
        if stoploss_triggered or regime_reversal:
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