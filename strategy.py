#!/usr/bin/env python3
"""
Experiment #172: 12h Primary + 1d/1w HTF — Simplified Mean Reversion + Trend Filter

Hypothesis: Previous strategies failed due to OVER-FILTERING (too many confluence requirements = 0 trades).
Research shows simpler signals with fewer filters generate more trades while maintaining edge.
This strategy uses:

1. 1w HMA(21) for MAJOR regime bias (bull/bear market detection)
2. 1d HMA(16) for intermediate trend direction
3. RSI(7) for entry timing (simpler than Connors, more responsive)
4. Bollinger Band %B for mean reversion extremes
5. ATR(14) for stoploss and vol adjustment

Key changes from failed experiments:
- REDUCED confluence requirements (score >= 1 instead of >= 2)
- SIMPLER RSI(7) instead of complex Connors RSI
- WEEKLY trend filter prevents fighting major moves
- More aggressive trade frequency safeguards (every 80 bars)
- Asymmetric sizing: 0.35 with trend, 0.20 against trend

Why this should work:
- 12h timeframe = 20-50 trades/year target (low fee drag)
- Weekly filter prevents catastrophic counter-trend trades
- RSI(7) is more responsive than RSI(14) for crypto volatility
- Fewer filters = more trades = better statistical significance
- Asymmetric sizing reduces risk when fighting trend

Timeframe: 12h (REQUIRED)
HTF: 1d and 1w via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.20-0.35 discrete
Stoploss: 2.5 * ATR(14) trailing
Target trades: 25-50/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_rsi_bb_hma_1d1w_simp_v1"
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
    """Calculate Bollinger Bands and %B."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    # Calculate %B: where price is within bands (0=lower, 1=upper)
    bb_range = upper - lower
    bb_range = np.where(bb_range == 0, 1e-10, bb_range)
    percent_b = (close - lower) / bb_range
    
    return upper, lower, sma, percent_b

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_hma_slope(hma_values, lookback=3):
    """Calculate HMA slope as percentage change over lookback."""
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
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d_16 = calculate_hma(df_1d['close'].values, 16)
    hma_1d_slope = calculate_hma_slope(hma_1d_16, 3)
    
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    hma_1w_slope = calculate_hma_slope(hma_1w_21, 2)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_16_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_16)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope)
    
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    hma_1w_slope_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_slope)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_7 = calculate_rsi(close, 7)
    rsi_14 = calculate_rsi(close, 14)
    
    bb_upper, bb_lower, bb_mid, percent_b = calculate_bollinger_bands(close, 20, 2.2)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE_WITH_TREND = 0.35
    BASE_SIZE_COUNTER = 0.20
    
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
        
        if np.isnan(hma_1d_16_aligned[i]) or np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(rsi_7[i]) or np.isnan(percent_b[i]):
            continue
        
        # === WEEKLY REGIME (Major trend - don't fight this) ===
        weekly_bullish = hma_1w_slope_aligned[i] > 0.5
        weekly_bearish = hma_1w_slope_aligned[i] < -0.5
        price_above_1w_hma = close[i] > hma_1w_21_aligned[i]
        price_below_1w_hma = close[i] < hma_1w_21_aligned[i]
        
        # === DAILY TREND (Intermediate direction) ===
        daily_bullish = hma_1d_slope_aligned[i] > 0.3
        daily_bearish = hma_1d_slope_aligned[i] < -0.3
        price_above_1d_hma = close[i] > hma_1d_16_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_16_aligned[i]
        
        # === RSI SIGNALS (Entry timing) ===
        rsi_oversold = rsi_7[i] < 30
        rsi_overbought = rsi_7[i] > 70
        rsi_extreme_low = rsi_7[i] < 20
        rsi_extreme_high = rsi_7[i] > 80
        rsi_neutral = 35 <= rsi_7[i] <= 65
        
        # === BOLLINGER BAND SIGNALS ===
        bb_extreme_low = percent_b[i] < 0.05  # Below lower band
        bb_extreme_high = percent_b[i] > 0.95  # Above upper band
        bb_lower_half = percent_b[i] < 0.4
        bb_upper_half = percent_b[i] > 0.6
        
        # === POSITION SIZING ===
        # Larger size when trading WITH weekly trend
        if weekly_bullish:
            long_size = BASE_SIZE_WITH_TREND
            short_size = BASE_SIZE_COUNTER
        elif weekly_bearish:
            long_size = BASE_SIZE_COUNTER
            short_size = BASE_SIZE_WITH_TREND
        else:
            long_size = BASE_SIZE_COUNTER
            short_size = BASE_SIZE_COUNTER
        
        # === ENTRY LOGIC (Simplified - fewer confluence requirements) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Simpler logic, more trades
        long_score = 0
        
        # Primary: RSI oversold + BB extreme (mean reversion)
        if rsi_oversold and bb_extreme_low:
            long_score += 3
        
        # Secondary: RSI very low alone (more trades)
        if rsi_extreme_low:
            long_score += 2
        
        # Tertiary: Pullback in uptrend
        if (weekly_bullish or daily_bullish) and rsi_7[i] < 40 and bb_lower_half:
            long_score += 2
        
        # Quaternary: Price below both HMA but RSI very low (deep pullback)
        if price_below_1d_hma and price_below_1w_hma and rsi_7[i] < 35:
            long_score += 1
        
        # Generate long signal with reduced threshold
        if long_score >= 2:
            new_signal = long_size
        elif long_score == 1 and bars_since_last_trade > 60:
            new_signal = long_size * 0.6
        
        # SHORT ENTRIES
        short_score = 0
        
        # Primary: RSI overbought + BB extreme
        if rsi_overbought and bb_extreme_high:
            short_score += 3
        
        # Secondary: RSI very high alone
        if rsi_extreme_high:
            short_score += 2
        
        # Tertiary: Rally in downtrend
        if (weekly_bearish or daily_bearish) and rsi_7[i] > 60 and bb_upper_half:
            short_score += 2
        
        # Quaternary: Price above both HMA but RSI very high (rally in bear)
        if price_above_1d_hma and price_above_1w_hma and rsi_7[i] > 65:
            short_score += 1
        
        # Generate short signal with reduced threshold
        if short_score >= 2:
            new_signal = -short_size
        elif short_score == 1 and bars_since_last_trade > 60:
            new_signal = -short_size * 0.6
        
        # === TRADE FREQUENCY SAFEGUARD (Critical for generating trades) ===
        # Force trade if no signal for 100 bars (~50 days on 12h)
        if bars_since_last_trade > 100 and new_signal == 0.0 and not in_position:
            if weekly_bullish and rsi_7[i] < 40:
                new_signal = long_size * 0.5
            elif weekly_bearish and rsi_7[i] > 60:
                new_signal = -short_size * 0.5
            elif rsi_7[i] < 25:
                new_signal = long_size * 0.4
            elif rsi_7[i] > 75:
                new_signal = -short_size * 0.4
        
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
        # Exit if weekly regime flips against position
        regime_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and weekly_bearish and price_below_1w_hma:
                regime_reversal = True
            if position_side < 0 and weekly_bullish and price_above_1w_hma:
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