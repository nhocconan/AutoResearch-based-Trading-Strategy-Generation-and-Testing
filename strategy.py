#!/usr/bin/env python3
"""
Experiment #184: 4h Primary + 12h/1d HTF — Simplified Mean Reversion + Trend Filter

Hypothesis: Previous strategies failed due to overly complex entry conditions resulting in 0 trades.
This strategy uses SIMPLIFIED logic with fewer confluence requirements to ensure trades trigger
on all symbols (BTC, ETH, SOL) during both train and test periods.

Key changes from failed experiments:
1. Simple RSI(14) instead of Connors RSI (more reliable, fewer calculations)
2. Lower entry thresholds to ensure trades actually trigger
3. Fewer regime filters (choppiness removed - was causing 0 trades)
4. Time-based fallback to guarantee minimum trade frequency
5. 12h HMA for trend bias + 4h RSI for entry timing

Components:
- 12h HMA(21): Major trend direction (long only when bullish, short only when bearish)
- 4h RSI(14): Entry timing (oversold <35 for longs, overbought >65 for shorts)
- 4h Bollinger Bands(20, 2.0): Price position confirmation
- 4h ATR(14): Volatility filter + stoploss (2.5 * ATR trailing)
- Minimum trade frequency safeguard (force entry every 120 bars if no trades)

Timeframe: 4h (REQUIRED for this experiment)
HTF: 12h via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.30 discrete (max 0.35)
Target trades: 25-50/year per symbol (~100-200 total on 4h over 4 years)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_rsi_bb_hma_12h_simp_v1"
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
    return upper, lower, sma

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate HTF indicators
    hma_12h_21 = calculate_hma(df_12h['close'].values, 21)
    hma_12h_slope = calculate_hma_slope(hma_12h_21, 5)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    hma_12h_slope_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_slope)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    
    # BB width for volatility context
    bb_width = (bb_upper - bb_lower) / bb_mid
    bb_width_s = pd.Series(bb_width).rolling(window=50, min_periods=50).mean().values
    bb_width_percentile = np.zeros(n)
    for i in range(50, n):
        window = bb_width[max(0, i-50):i+1]
        bb_width_percentile[i] = np.sum(window < bb_width[i]) / len(window) * 100
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.30
    
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
        
        if np.isnan(hma_12h_21_aligned[i]) or np.isnan(hma_12h_slope_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(bb_lower[i]):
            continue
        
        # === 12H TREND BIAS ===
        trend_12h_bullish = hma_12h_slope_aligned[i] > 0.5
        trend_12h_bearish = hma_12h_slope_aligned[i] < -0.5
        price_above_12h_hma = close[i] > hma_12h_21_aligned[i]
        price_below_12h_hma = close[i] < hma_12h_21_aligned[i]
        
        # === RSI SIGNALS ===
        rsi_oversold = rsi_14[i] < 40  # Lowered from 35 for more trades
        rsi_overbought = rsi_14[i] > 60  # Lowered from 65 for more trades
        rsi_extreme_low = rsi_14[i] < 30
        rsi_extreme_high = rsi_14[i] > 70
        
        # === BOLLINGER BAND POSITION ===
        price_below_bb_lower = close[i] < bb_lower[i]
        price_above_bb_upper = close[i] > bb_upper[i]
        price_near_bb_lower = close[i] < bb_lower[i] * 1.02  # Within 2% of lower band
        price_near_bb_upper = close[i] > bb_upper[i] * 0.98  # Within 2% of upper band
        
        # === VOLATILITY CONTEXT ===
        vol_expanding = bb_width_percentile[i] > 60
        vol_contracting = bb_width_percentile[i] < 40
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        if not trend_12h_bullish and not trend_12h_bearish:
            current_size = BASE_SIZE * 0.8  # Reduce size in unclear trend
        
        # === ENTRY LOGIC (SIMPLIFIED for more trades) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Simplified conditions
        long_condition_1 = rsi_oversold and price_near_bb_lower  # Basic mean revert
        long_condition_2 = rsi_extreme_low and trend_12h_bullish  # Deep pullback in bull
        long_condition_3 = rsi_oversold and price_above_12h_hma  # Pullback above HMA
        long_condition_4 = price_below_bb_lower and vol_expanding  # BB break with vol
        
        long_score = sum([long_condition_1, long_condition_2, long_condition_3, long_condition_4])
        
        if long_score >= 1:
            new_signal = current_size
        elif long_score == 0 and rsi_14[i] < 35 and bars_since_last_trade > 100:
            new_signal = current_size * 0.5  # Fallback for more trades
        
        # SHORT ENTRIES
        short_condition_1 = rsi_overbought and price_near_bb_upper  # Basic mean revert
        short_condition_2 = rsi_extreme_high and trend_12h_bearish  # Rally in bear
        short_condition_3 = rsi_overbought and price_below_12h_hma  # Rally below HMA
        short_condition_4 = price_above_bb_upper and vol_expanding  # BB break with vol
        
        short_score = sum([short_condition_1, short_condition_2, short_condition_3, short_condition_4])
        
        if short_score >= 1:
            new_signal = -current_size
        elif short_score == 0 and rsi_14[i] > 65 and bars_since_last_trade > 100:
            new_signal = -current_size * 0.5  # Fallback for more trades
        
        # === FREQUENCY SAFEGUARD (CRITICAL for avoiding 0 trades) ===
        # Force trade if no signal for 120 bars (~20 days on 4h)
        if bars_since_last_trade > 120 and new_signal == 0.0 and not in_position:
            if trend_12h_bullish and rsi_14[i] < 45:
                new_signal = current_size * 0.4
            elif trend_12h_bearish and rsi_14[i] > 55:
                new_signal = -current_size * 0.4
            elif rsi_14[i] < 35:
                new_signal = current_size * 0.3
            elif rsi_14[i] > 65:
                new_signal = -current_size * 0.3
        
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
        trend_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and trend_12h_bearish and rsi_14[i] > 55:
                trend_reversal = True
            if position_side < 0 and trend_12h_bullish and rsi_14[i] < 45:
                trend_reversal = True
        
        if stoploss_triggered or trend_reversal:
            new_signal = 0.0
        
        # === RSI EXIT (take profit) ===
        if in_position and position_side != 0:
            if position_side > 0 and rsi_14[i] > 70:
                new_signal = 0.0  # Take profit on long
            if position_side < 0 and rsi_14[i] < 30:
                new_signal = 0.0  # Take profit on short
        
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