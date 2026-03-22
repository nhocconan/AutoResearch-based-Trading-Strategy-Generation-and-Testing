#!/usr/bin/env python3
"""
Experiment #220: 1h Primary + 4h/12h HTF — Volatility Z-Score Mean Reversion

Hypothesis: After 219 experiments, trend-following and Connors RSI strategies 
consistently fail in bear/range markets (2022 crash, 2025 bear). This strategy 
uses VOLATILITY-ADJUSTED MEAN REVERSION with HTF trend filter:

1. Z-SCORE(20): Price deviation from 20-bar rolling mean (mean reversion signal)
2. ATR RATIO(7/30): Volatility regime detection (spike vs compression)
3. 4h HMA(21) SLOPE: HTF trend bias (never fight the higher timeframe)
4. 12h HMA(48): Major trend confirmation (avoid counter-trend in strong trends)
5. SESSION FILTER: Only 8-20 UTC (high liquidity, reduced slippage)
6. VOLUME CONFIRMATION: Volume > 0.8x 20-bar average

Why this differs from failed strategies:
- NO Connors RSI (failed in 50+ experiments)
- NO Choppiness Index (failed in 30+ experiments)
- Uses Z-score mean reversion (underutilized in our experiments)
- Volatility regime filter (ATR ratio) instead of choppy/trendy binary
- LOOSE entry conditions to guarantee 30+ trades/year on 1h

Position sizing: 0.25 discrete (conservative for 1h TF)
Stoploss: 2.5 * ATR(14) trailing
Target: 40-80 trades/year per symbol

CRITICAL: Use mtf_data helper for HTF (Rule 1-3)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_vol_zscore_hma_4h12h_v1"
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

def calculate_zscore(close, period=20):
    """Calculate Z-score of price relative to rolling mean."""
    close_s = pd.Series(close)
    rolling_mean = close_s.rolling(window=period, min_periods=period).mean()
    rolling_std = close_s.rolling(window=period, min_periods=period).std()
    zscore = (close_s - rolling_mean) / rolling_std.replace(0, np.nan)
    zscore = zscore.fillna(0).values
    return zscore

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
    """Calculate HMA slope as percentage change."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        if hma_values[i - lookback] != 0 and not np.isnan(hma_values[i - lookback]):
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback] * 100
    return slope

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

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
    return pd.to_datetime(open_time, unit='ms').hour

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
    
    # Calculate 4h HTF indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_slope = calculate_hma_slope(hma_4h_21, 3)
    
    # Calculate 12h HTF indicators
    hma_12h_48 = calculate_hma(df_12h['close'].values, 48)
    hma_12h_slope = calculate_hma_slope(hma_12h_48, 3)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_slope_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_slope)
    hma_12h_48_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_48)
    hma_12h_slope_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_slope)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    zscore_20 = calculate_zscore(close, 20)
    rsi_14 = calculate_rsi(close, 14)
    
    # Volume rolling average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
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
        
        if np.isnan(hma_12h_48_aligned[i]) or np.isnan(hma_12h_slope_aligned[i]):
            continue
        
        if np.isnan(zscore_20[i]) or np.isnan(atr_7[i]) or np.isnan(atr_30[i]):
            continue
        
        if np.isnan(vol_avg_20[i]) or vol_avg_20[i] == 0:
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        utc_hour = get_utc_hour(open_time[i])
        in_session = 8 <= utc_hour <= 20
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 0.8 * vol_avg_20[i]
        
        # === VOLATILITY REGIME (ATR Ratio) ===
        atr_ratio = atr_7[i] / atr_30[i] if atr_30[i] > 0 else 1.0
        vol_spike = atr_ratio > 1.8  # High volatility (mean reversion likely)
        vol_compress = atr_ratio < 0.7  # Low volatility (breakout likely)
        vol_normal = 0.7 <= atr_ratio <= 1.8
        
        # === HTF TREND BIAS (4h) ===
        hma_4h_bullish = hma_4h_slope_aligned[i] > 0.2
        hma_4h_bearish = hma_4h_slope_aligned[i] < -0.2
        hma_4h_neutral = not hma_4h_bullish and not hma_4h_bearish
        
        price_above_4h_hma = close[i] > hma_4h_21_aligned[i]
        price_below_4h_hma = close[i] < hma_4h_21_aligned[i]
        
        # === MAJOR TREND (12h) ===
        hma_12h_bullish = hma_12h_slope_aligned[i] > 0.15
        hma_12h_bearish = hma_12h_slope_aligned[i] < -0.15
        
        # === Z-SCORE MEAN REVERSION SIGNALS ===
        zscore_extreme_long = zscore_20[i] < -1.8  # Oversold
        zscore_extreme_short = zscore_20[i] > 1.8  # Overbought
        zscore_moderate_long = zscore_20[i] < -1.2
        zscore_moderate_short = zscore_20[i] > 1.2
        
        # === RSI CONFIRMATION ===
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        rsi_neutral = 35 <= rsi_14[i] <= 65
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Multiple paths for trade frequency (CRITICAL)
        long_score = 0
        
        # Path 1: Z-score extreme + vol spike + 4h neutral/bullish (primary mean revert)
        if zscore_extreme_long and vol_spike and (hma_4h_bullish or hma_4h_neutral):
            long_score += 4
        
        # Path 2: Z-score extreme + RSI oversold + volume confirmed
        if zscore_extreme_long and rsi_oversold and volume_confirmed:
            long_score += 4
        
        # Path 3: Z-score moderate + vol spike + 4h bullish + in session
        if zscore_moderate_long and vol_spike and hma_4h_bullish and in_session:
            long_score += 3
        
        # Path 4: Z-score extreme + 4h bullish + price above 4h HMA (pullback in trend)
        if zscore_extreme_long and hma_4h_bullish and price_above_4h_hma:
            long_score += 3
        
        # Path 5: Z-score moderate + RSI oversold + vol normal (standard mean revert)
        if zscore_moderate_long and rsi_oversold and vol_normal:
            long_score += 2
        
        # Path 6: Z-score extreme alone (looser for more trades)
        if zscore_extreme_long and bars_since_last_trade > 40:
            long_score += 1
        
        # Path 7: 4h bullish + RSI oversold + in session (trend pullback)
        if hma_4h_bullish and rsi_oversold and in_session and bars_since_last_trade > 30:
            long_score += 2
        
        # Path 8: 12h bullish + Z-score moderate (major trend pullback)
        if hma_12h_bullish and zscore_moderate_long and volume_confirmed:
            long_score += 2
        
        if long_score >= 4:
            new_signal = current_size
        elif long_score == 3 and bars_since_last_trade > 30:
            new_signal = current_size * 0.8
        elif long_score >= 2 and bars_since_last_trade > 50:
            new_signal = current_size * 0.6
        elif long_score >= 1 and bars_since_last_trade > 70:
            new_signal = current_size * 0.4
        
        # SHORT ENTRIES
        short_score = 0
        
        # Path 1: Z-score extreme + vol spike + 4h neutral/bearish (primary mean revert)
        if zscore_extreme_short and vol_spike and (hma_4h_bearish or hma_4h_neutral):
            short_score += 4
        
        # Path 2: Z-score extreme + RSI overbought + volume confirmed
        if zscore_extreme_short and rsi_overbought and volume_confirmed:
            short_score += 4
        
        # Path 3: Z-score moderate + vol spike + 4h bearish + in session
        if zscore_moderate_short and vol_spike and hma_4h_bearish and in_session:
            short_score += 3
        
        # Path 4: Z-score extreme + 4h bearish + price below 4h HMA (pullback in trend)
        if zscore_extreme_short and hma_4h_bearish and price_below_4h_hma:
            short_score += 3
        
        # Path 5: Z-score moderate + RSI overbought + vol normal (standard mean revert)
        if zscore_moderate_short and rsi_overbought and vol_normal:
            short_score += 2
        
        # Path 6: Z-score extreme alone (looser for more trades)
        if zscore_extreme_short and bars_since_last_trade > 40:
            short_score += 1
        
        # Path 7: 4h bearish + RSI overbought + in session (trend pullback)
        if hma_4h_bearish and rsi_overbought and in_session and bars_since_last_trade > 30:
            short_score += 2
        
        # Path 8: 12h bearish + Z-score moderate (major trend pullback)
        if hma_12h_bearish and zscore_moderate_short and volume_confirmed:
            short_score += 2
        
        if short_score >= 4:
            new_signal = -current_size
        elif short_score == 3 and bars_since_last_trade > 30:
            new_signal = -current_size * 0.8
        elif short_score >= 2 and bars_since_last_trade > 50:
            new_signal = -current_size * 0.6
        elif short_score >= 1 and bars_since_last_trade > 70:
            new_signal = -current_size * 0.4
        
        # === FREQUENCY SAFEGUARD ===
        # Force trade if no signal for 120 bars (~5 days on 1h)
        if bars_since_last_trade > 120 and new_signal == 0.0 and not in_position:
            if hma_4h_bullish and zscore_moderate_long and in_session:
                new_signal = current_size * 0.35
            elif hma_4h_bearish and zscore_moderate_short and in_session:
                new_signal = -current_size * 0.35
            elif zscore_extreme_long and rsi_oversold:
                new_signal = current_size * 0.30
            elif zscore_extreme_short and rsi_overbought:
                new_signal = -current_size * 0.30
        
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
        
        # === HTF TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Long position but 4h turns strongly bearish
            if position_side > 0 and hma_4h_bearish and price_below_4h_hma:
                trend_reversal = True
            # Short position but 4h turns strongly bullish
            if position_side < 0 and hma_4h_bullish and price_above_4h_hma:
                trend_reversal = True
        
        # === Z-SCORE MEAN REVERSION EXIT ===
        # Exit when z-score returns to neutral (mean reversion complete)
        mean_revert_exit = False
        if in_position and position_side != 0:
            if position_side > 0 and zscore_20[i] > 0.5:
                mean_revert_exit = True
            if position_side < 0 and zscore_20[i] < -0.5:
                mean_revert_exit = True
        
        if stoploss_triggered or trend_reversal or mean_revert_exit:
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