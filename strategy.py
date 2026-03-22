#!/usr/bin/env python3
"""
Experiment #020: 1h Mean Reversion with 4h/12h Trend Filter

Hypothesis: After 19 failed experiments with pure trend-following and regime-switching,
switch to MEAN REVERSION within HTF trend direction. This addresses the bear/range
market conditions of 2025+ where trend strategies consistently fail.

Key components:
1. 4h HMA(21) - determines primary trend direction (only trade WITH HTF trend)
2. 12h HMA(48) - confirms major trend bias (avoid counter-trend mean reversion)
3. 1h RSI(14) - mean reversion trigger (RSI<30 long, RSI>70 short)
4. 1h Bollinger Bands(20,2.0) - price at bands confirms oversold/overbought
5. Volume filter - volume > 0.8x 20-period average (confirms move significance)
6. Session filter - only 8-20 UTC (high liquidity hours, avoids Asian session noise)
7. ATR(14) stoploss - 2.5 ATR trailing stop to limit losses

Why this differs from failed attempts:
- NOT pure trend-following (failed in bear market)
- NOT complex regime detection (Choppiness/Fisher all failed)
- NOT too many filters (previous strategies had 0 trades)
- Mean reversion WITH trend filter = best of both worlds
- 1h timeframe with 4h/12h confirmation = fewer trades, better timing

Timeframe: 1h (REQUIRED)
HTF: 4h and 12h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete (smaller for lower TF, reduces fee impact)
Target trades: 40-80/year (strict entry conditions)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_meanrev_rsi_bb_4h_12h_hma_session_v1"
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
    loss = (-delta).where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    
    return upper.values, lower.values, sma.values

def calculate_volume_avg(volume, period=20):
    """Calculate rolling average volume."""
    vol_s = pd.Series(volume)
    return vol_s.rolling(window=period, min_periods=period).mean().values

def get_hour_from_open_time(open_time):
    """Extract hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
    hours = (open_time // (1000 * 60 * 60)) % 24
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate HTF indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_12h_48 = calculate_hma(df_12h['close'].values, 48)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_12h_48_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_48)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_sma = calculate_bollinger_bands(close, 20, 2.0)
    vol_avg_20 = calculate_volume_avg(volume, 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25  # Conservative for 1h timeframe
    
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
        
        if np.isnan(hma_12h_48_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]):
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        
        if np.isnan(vol_avg_20[i]) or vol_avg_20[i] == 0:
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        hour = get_hour_from_open_time(open_time[i])
        in_session = 8 <= hour <= 20
        
        # === VOLUME FILTER ===
        volume_confirmed = volume[i] > 0.8 * vol_avg_20[i]
        
        # === 4H TREND DIRECTION ===
        trend_4h_bullish = close[i] > hma_4h_21_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_21_aligned[i]
        
        # === 12H TREND CONFIRMATION ===
        trend_12h_bullish = close[i] > hma_12h_48_aligned[i]
        trend_12h_bearish = close[i] < hma_12h_48_aligned[i]
        
        # === RSI MEAN REVERSION SIGNALS ===
        rsi_oversold = rsi_14[i] < 30
        rsi_overbought = rsi_14[i] > 70
        
        # === BOLLINGER BAND CONFIRMATION ===
        at_lower_band = close[i] <= bb_lower[i] * 1.002  # Within 0.2% of lower band
        at_upper_band = close[i] >= bb_upper[i] * 0.998  # Within 0.2% of upper band
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG ENTRY: Mean reversion WITH uptrend
        # Need: 4h bullish + (12h bullish OR neutral) + RSI oversold + at lower band + volume + session
        long_trend_confirmed = trend_4h_bullish and (trend_12h_bullish or not trend_12h_bearish)
        long_mean_reversion = rsi_oversold and at_lower_band
        
        if long_trend_confirmed and long_mean_reversion and volume_confirmed and in_session:
            new_signal = BASE_SIZE
        
        # SHORT ENTRY: Mean reversion WITH downtrend
        # Need: 4h bearish + (12h bearish OR neutral) + RSI overbought + at upper band + volume + session
        short_trend_confirmed = trend_4h_bearish and (trend_12h_bearish or not trend_12h_bullish)
        short_mean_reversion = rsi_overbought and at_upper_band
        
        if short_trend_confirmed and short_mean_reversion and volume_confirmed and in_session:
            new_signal = -BASE_SIZE
        
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
        
        # === MEAN REVERSION EXIT (RSI returns to neutral) ===
        mean_reversion_exit = False
        if in_position and position_side != 0:
            # Exit long when RSI returns above 50 (mean restored)
            if position_side > 0 and rsi_14[i] > 55:
                mean_reversion_exit = True
            # Exit short when RSI returns below 50 (mean restored)
            if position_side < 0 and rsi_14[i] < 45:
                mean_reversion_exit = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal_exit = False
        if in_position and position_side != 0:
            # Exit long if 4h trend turns bearish
            if position_side > 0 and trend_4h_bearish:
                trend_reversal_exit = True
            # Exit short if 4h trend turns bullish
            if position_side < 0 and trend_4h_bullish:
                trend_reversal_exit = True
        
        # Apply stoploss or exit conditions
        if stoploss_triggered or mean_reversion_exit or trend_reversal_exit:
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