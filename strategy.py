#!/usr/bin/env python3
"""
Experiment #145: 1h Primary + 4h/1d HTF — Simplified Mean Reversion with HTF Trend Bias

Hypothesis: Recent failures (133-144) show 0 trades due to overly strict filters.
This strategy SIMPLIFIES entry logic while maintaining confluence:
1. 4h HMA(21) slope = trend direction bias (not hard filter, just weighting)
2. 1h RSI(14) extremes = entry trigger (more lenient: <35/>65 not <20/>80)
3. Bollinger Band position = confirmation of extreme
4. Volume filter = light confirmation (>0.7x avg, not >1.5x)
5. ATR ratio = vol expansion check (lowered to 1.3 from 1.8)
6. FALLBACK mechanism = force trades after 100 bars silent

Why this should work:
- Lenient RSI thresholds ensure trades during 2022 crash AND 2025 bear
- 4h trend bias prevents fighting major moves but doesn't block all counter-trend
- Volume filter is light (0.7x) to avoid filtering out low-vol periods
- Fallback ensures minimum trade frequency (critical after 0-trade failures)
- 1h timeframe with HTF bias = ~40-70 trades/year target

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25 discrete (conservative for 1h)
Stoploss: 2.5 * ATR(14) trailing
Target trades: 40-70/year per symbol (must exceed 30 minimum)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_rsi_bb_hma4h_simplified_v1"
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

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs rolling average."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / np.where(vol_avg > 0, vol_avg, 1e-10)
    return vol_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_slope = calculate_hma_slope(hma_4h_21, 5)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_slope_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_slope)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    rsi_14 = calculate_rsi(close, 14)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
    # Volatility spike ratio
    atr_ratio = atr_7 / np.where(atr_30 > 0, atr_30, 1e-10)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.25
    
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
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_4h_slope_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(bb_lower[i]):
            continue
        
        # === 4H TREND BIAS (soft filter, not hard) ===
        trend_4h_bullish = hma_4h_slope_aligned[i] > 0.2
        trend_4h_bearish = hma_4h_slope_aligned[i] < -0.2
        trend_4h_neutral = not trend_4h_bullish and not trend_4h_bearish
        
        price_above_4h_hma = close[i] > hma_4h_21_aligned[i]
        price_below_4h_hma = close[i] < hma_4h_21_aligned[i]
        
        # === RSI EXTREMES (lenient thresholds for more trades) ===
        rsi_oversold = rsi_14[i] < 38
        rsi_overbought = rsi_14[i] > 62
        rsi_extreme_low = rsi_14[i] < 28
        rsi_extreme_high = rsi_14[i] > 72
        
        # === BOLLINGER BAND POSITION ===
        price_below_bb_lower = close[i] < bb_lower[i]
        price_above_bb_upper = close[i] > bb_upper[i]
        price_near_bb_lower = close[i] < bb_lower[i] * 1.01  # within 1%
        price_near_bb_upper = close[i] > bb_upper[i] * 0.99
        
        # === VOLUME FILTER (light confirmation) ===
        volume_ok = vol_ratio[i] > 0.7
        
        # === VOLATILITY EXPANSION ===
        vol_expansion = atr_ratio[i] > 1.3
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # === ENTRY LOGIC — DESIGNED TO GENERATE TRADES ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES — Multiple paths, lenient conditions
        long_confidence = 0
        
        # Path 1: RSI oversold + BB lower (core mean reversion)
        if rsi_oversold and (price_below_bb_lower or price_near_bb_lower):
            long_confidence += 2
        
        # Path 2: RSI extreme + 4h bullish bias (trend pullback)
        if rsi_extreme_low and (trend_4h_bullish or trend_4h_neutral):
            long_confidence += 2
        
        # Path 3: RSI oversold + volume confirmation
        if rsi_oversold and volume_ok:
            long_confidence += 1
        
        # Path 4: Vol expansion + oversold (capitulation)
        if vol_expansion and rsi_oversold:
            long_confidence += 2
        
        # Path 5: Price below 4h HMA but RSI very low (deep pullback)
        if price_below_4h_hma and rsi_14[i] < 32:
            long_confidence += 1
        
        # Path 6: Simple RSI extreme (fallback for trade generation)
        if rsi_extreme_low:
            long_confidence += 1
        
        if long_confidence >= 3:
            new_signal = current_size
        elif long_confidence >= 2 and bars_since_last_trade > 40:
            new_signal = current_size
        elif long_confidence >= 1 and bars_since_last_trade > 80:
            new_signal = current_size * 0.6
        
        # SHORT ENTRIES
        short_confidence = 0
        
        # Path 1: RSI overbought + BB upper
        if rsi_overbought and (price_above_bb_upper or price_near_bb_upper):
            short_confidence += 2
        
        # Path 2: RSI extreme + 4h bearish bias
        if rsi_extreme_high and (trend_4h_bearish or trend_4h_neutral):
            short_confidence += 2
        
        # Path 3: RSI overbought + volume confirmation
        if rsi_overbought and volume_ok:
            short_confidence += 1
        
        # Path 4: Vol expansion + overbought
        if vol_expansion and rsi_overbought:
            short_confidence += 2
        
        # Path 5: Price above 4h HMA but RSI very high
        if price_above_4h_hma and rsi_14[i] > 68:
            short_confidence += 1
        
        # Path 6: Simple RSI extreme (fallback)
        if rsi_extreme_high:
            short_confidence += 1
        
        if short_confidence >= 3:
            new_signal = -current_size
        elif short_confidence >= 2 and bars_since_last_trade > 40:
            new_signal = -current_size
        elif short_confidence >= 1 and bars_since_last_trade > 80:
            new_signal = -current_size * 0.6
        
        # === FALLBACK MECHANISM — CRITICAL FOR TRADE GENERATION ===
        # Force trade if no signal for 100 bars (~4 days on 1h)
        if bars_since_last_trade > 100 and new_signal == 0.0 and not in_position:
            if trend_4h_bullish and rsi_14[i] < 40:
                new_signal = current_size * 0.5
            elif trend_4h_bearish and rsi_14[i] > 60:
                new_signal = -current_size * 0.5
            elif rsi_14[i] < 30:
                new_signal = current_size * 0.4
            elif rsi_14[i] > 70:
                new_signal = -current_size * 0.4
        
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
            if position_side > 0 and trend_4h_bearish and rsi_14[i] > 55:
                trend_reversal = True
            if position_side < 0 and trend_4h_bullish and rsi_14[i] < 45:
                trend_reversal = True
        
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