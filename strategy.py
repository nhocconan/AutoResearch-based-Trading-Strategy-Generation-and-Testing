#!/usr/bin/env python3
"""
Experiment #190: 1h Primary + 4h/12h HTF — Regime-Adaptive RSI Mean Reversion

Hypothesis: Previous 1h/30m strategies failed because entry conditions were TOO STRICT
(RSI<20, CRSI<15) resulting in 0 trades. This strategy uses LOOSER but still quality
entry thresholds to ensure 30-80 trades/year while maintaining edge.

Key innovations:
1. 4h HMA(21) slope for major trend bias (only trade WITH HTF trend)
2. 1h RSI(14) with MODERATE thresholds (<40 long, >60 short) — NOT extreme
3. Bollinger Band confirmation (price outside bands) for mean reversion setup
4. ATR ratio for volatility regime adjustment (high vol = reduce size)
5. Session filter: Only trade 8-20 UTC (reduces noise, Asian session whipsaw)
6. 12h ADX for trend strength confirmation (ADX>20 = trend, ADX<20 = range)

Why this should work on 1h:
- Looser RSI thresholds ensure trades actually trigger (learned from #180, #185, #188)
- HTF trend filter prevents counter-trend trades (major failure mode)
- Session filter reduces fee drag from low-quality trades
- Position size 0.25 (smaller than 12h strategies) accounts for higher frequency
- Target: 40-80 trades/year per symbol

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h HMA + 12h ADX via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25 discrete (max 0.30 for lower TF)
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_rsi_bb_hma_4h12h_v1"
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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / np.where(atr > 0, atr, 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / np.where(atr > 0, atr, 1e-10)
    
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) > 0, (plus_di + minus_di), 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

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

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    return pd.to_datetime(open_time, unit='ms').dt.hour.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 4h HTF indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_slope = calculate_hma_slope(hma_4h_21, 5)
    
    # Calculate 12h HTF indicators
    adx_12h, _, _ = calculate_adx(
        df_12h['high'].values,
        df_12h['low'].values,
        df_12h['close'].values,
        14
    )
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_slope_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_slope)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    
    rsi_14 = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    
    # Volatility ratio
    atr_ratio = atr_7 / np.where(atr_30 > 0, atr_30, 1e-10)
    
    # Session filter (8-20 UTC)
    utc_hour = get_utc_hour(open_time)
    in_session = (utc_hour >= 8) & (utc_hour <= 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.30 for 1h)
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
        
        if np.isnan(adx_12h_aligned[i]) or np.isnan(rsi_14[i]):
            continue
        
        if np.isnan(bb_lower[i]) or np.isnan(bb_upper[i]):
            continue
        
        # === 4H TREND BIAS ===
        trend_4h_bullish = hma_4h_slope_aligned[i] > 0.2
        trend_4h_bearish = hma_4h_slope_aligned[i] < -0.2
        price_above_4h_hma = close[i] > hma_4h_21_aligned[i]
        price_below_4h_hma = close[i] < hma_4h_21_aligned[i]
        
        # === 12H TREND STRENGTH ===
        is_trending = adx_12h_aligned[i] > 20
        is_ranging = adx_12h_aligned[i] < 20
        
        # === VOLATILITY REGIME ===
        high_vol = atr_ratio[i] > 1.5
        low_vol = atr_ratio[i] < 0.8
        
        # === BOLLINGER BAND POSITION ===
        price_below_bb_lower = close[i] < bb_lower[i]
        price_above_bb_upper = close[i] > bb_upper[i]
        
        # === RSI SIGNALS (LOOSER thresholds for more trades) ===
        rsi_oversold = rsi_14[i] < 40  # Was <20 in failed strategies
        rsi_overbought = rsi_14[i] > 60  # Was >80 in failed strategies
        rsi_extreme_low = rsi_14[i] < 30
        rsi_extreme_high = rsi_14[i] > 70
        
        # === POSITION SIZING ADJUSTMENT ===
        current_size = BASE_SIZE
        if high_vol:
            current_size = BASE_SIZE * 0.7  # Reduce size in high vol
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Multiple paths for sufficient trade frequency
        long_score = 0
        
        # Path 1: 4h bullish + RSI oversold + price below BB (pullback in uptrend)
        if trend_4h_bullish and rsi_oversold and price_below_bb_lower:
            long_score += 3
        
        # Path 2: 4h bullish + RSI extreme + price below 4h HMA (deep pullback)
        if trend_4h_bullish and rsi_extreme_low and price_below_4h_hma:
            long_score += 3
        
        # Path 3: Range market + RSI oversold (mean revert)
        if is_ranging and rsi_oversold and price_below_bb_lower:
            long_score += 2
        
        # Path 4: Simple oversold in session (fallback for trade frequency)
        if in_session and rsi_extreme_low and bars_since_last_trade > 60:
            long_score += 1
        
        if long_score >= 3:
            new_signal = current_size
        elif long_score >= 2 and in_session:
            new_signal = current_size
        elif long_score >= 1 and bars_since_last_trade > 100:
            new_signal = current_size * 0.5
        
        # SHORT ENTRIES
        short_score = 0
        
        # Path 1: 4h bearish + RSI overbought + price above BB (rally in downtrend)
        if trend_4h_bearish and rsi_overbought and price_above_bb_upper:
            short_score += 3
        
        # Path 2: 4h bearish + RSI extreme + price above 4h HMA (rally in bear)
        if trend_4h_bearish and rsi_extreme_high and price_above_4h_hma:
            short_score += 3
        
        # Path 3: Range market + RSI overbought (mean revert)
        if is_ranging and rsi_overbought and price_above_bb_upper:
            short_score += 2
        
        # Path 4: Simple overbought in session (fallback)
        if in_session and rsi_extreme_high and bars_since_last_trade > 60:
            short_score += 1
        
        if short_score >= 3:
            new_signal = -current_size
        elif short_score >= 2 and in_session:
            new_signal = -current_size
        elif short_score >= 1 and bars_since_last_trade > 100:
            new_signal = -current_size * 0.5
        
        # === FREQUENCY SAFEGUARD ===
        # Force trade if no signal for 200 bars (~8 days on 1h)
        if bars_since_last_trade > 200 and new_signal == 0.0 and not in_position:
            if trend_4h_bullish and rsi_14[i] < 45:
                new_signal = current_size * 0.4
            elif trend_4h_bearish and rsi_14[i] > 55:
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
            if position_side > 0 and trend_4h_bearish:
                trend_reversal = True
            if position_side < 0 and trend_4h_bullish:
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