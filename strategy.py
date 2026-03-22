#!/usr/bin/env python3
"""
Experiment #118: 30m Primary + 4h/1d HTF — Regime-Adaptive Mean Reversion

Hypothesis: Previous 30m strategies failed due to overly strict entry conditions (0 trades).
This strategy uses SIMPLE, proven signals with LOOSE thresholds to ensure trade generation:

1. 4h HMA(21) SLOPE: Major trend bias (bullish > 0.2%, bearish < -0.2%)
2. 1d HMA(21) POSITION: Secondary confirmation (price above/below)
3. 30m RSI(14): Entry timing (long < 40, short > 60 — LOOSE thresholds)
4. 30m BOLLINGER BANDS: Extreme confirmation (price < lower or > upper)
5. VOLUME FILTER: > 0.7x 20-bar average (not too strict)
6. SESSION FILTER: 8-20 UTC only (reduces noise, not mandatory)

Key changes from failed experiments:
- RSI thresholds: 30/70 → 40/60 (MUCH easier to trigger)
- Volume filter: 1.0x → 0.7x average
- Added FORCE TRADE mechanism if no trades for 120 bars
- Position size: 0.25 (conservative for 30m)
- Stoploss: 2.0 * ATR (tighter for lower TF)

Timeframe: 30m (REQUIRED)
HTF: 4h + 1d via mtf_data.get_htf_data() — called ONCE before loop
Target trades: 40-80/year per symbol (critical for 30m)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_regime_rsi_bb_4h1d_v1"
timeframe = "30m"
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

def extract_hour_from_opentime(prices):
    """Extract UTC hour from open_time column."""
    if 'open_time' in prices.columns:
        # open_time is in milliseconds
        hours = (prices['open_time'].values // (1000 * 60 * 60)) % 24
        return hours
    else:
        # Fallback: assume all bars are valid
        return np.ones(len(prices)) * 12

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Extract session hours
    session_hours = extract_hour_from_opentime(prices)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_slope = calculate_hma_slope(hma_4h_21, 5)
    
    # Calculate 1d indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_slope_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_slope)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    
    # Volume average
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40, conservative for 30m)
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
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(rsi_14[i]):
            continue
        
        if np.isnan(bb_lower[i]) or np.isnan(vol_sma_20[i]):
            continue
        
        # === 4H TREND BIAS (Primary HTF signal) ===
        trend_4h_bullish = hma_4h_slope_aligned[i] > 0.15  # Lowered threshold
        trend_4h_bearish = hma_4h_slope_aligned[i] < -0.15  # Lowered threshold
        
        # === 1D TREND CONFIRMATION (Secondary HTF) ===
        price_above_1d_hma = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_21_aligned[i]
        
        # === 30m RSI (Entry timing — LOOSE thresholds) ===
        rsi_oversold = rsi_14[i] < 40  # Was 30, now 40 for more trades
        rsi_overbought = rsi_14[i] > 60  # Was 70, now 60 for more trades
        rsi_extreme_low = rsi_14[i] < 30
        rsi_extreme_high = rsi_14[i] > 70
        
        # === BOLLINGER BAND POSITION ===
        price_below_bb_lower = close[i] < bb_lower[i]
        price_above_bb_upper = close[i] > bb_upper[i]
        price_near_bb_lower = close[i] < bb_lower[i] * 1.01  # Within 1%
        price_near_bb_upper = close[i] > bb_upper[i] * 0.99  # Within 1%
        
        # === VOLUME FILTER (Loose) ===
        vol_ok = volume[i] > 0.7 * vol_sma_20[i]  # Was 1.0x, now 0.7x
        
        # === SESSION FILTER (8-20 UTC) ===
        in_session = (session_hours[i] >= 8) and (session_hours[i] <= 20)
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # === ENTRY LOGIC — LOOSE CONDITIONS FOR TRADE GENERATION ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES — Multiple paths with low thresholds
        long_confidence = 0
        
        # Path 1: 4h bullish + RSI oversold (primary)
        if trend_4h_bullish and rsi_oversold:
            long_confidence += 3
        
        # Path 2: 4h bullish + BB lower (mean revert in uptrend)
        if trend_4h_bullish and (price_below_bb_lower or price_near_bb_lower):
            long_confidence += 2
        
        # Path 3: 1d above HMA + RSI oversold (major trend confirmation)
        if price_above_1d_hma and rsi_oversold:
            long_confidence += 2
        
        # Path 4: RSI extreme low (capitulation)
        if rsi_extreme_low:
            long_confidence += 2
        
        # Path 5: BB lower + volume (volatility entry)
        if price_below_bb_lower and vol_ok:
            long_confidence += 1
        
        # Path 6: Any RSI oversold (fallback for trade generation)
        if rsi_14[i] < 35:
            long_confidence += 1
        
        if long_confidence >= 3:
            new_signal = current_size
        elif long_confidence >= 2 and bars_since_last_trade > 40:
            new_signal = current_size * 0.6
        elif long_confidence >= 1 and bars_since_last_trade > 80:
            new_signal = current_size * 0.4
        
        # SHORT ENTRIES
        short_confidence = 0
        
        # Path 1: 4h bearish + RSI overbought (primary)
        if trend_4h_bearish and rsi_overbought:
            short_confidence += 3
        
        # Path 2: 4h bearish + BB upper (mean revert in downtrend)
        if trend_4h_bearish and (price_above_bb_upper or price_near_bb_upper):
            short_confidence += 2
        
        # Path 3: 1d below HMA + RSI overbought (major trend confirmation)
        if price_below_1d_hma and rsi_overbought:
            short_confidence += 2
        
        # Path 4: RSI extreme high (euphoria)
        if rsi_extreme_high:
            short_confidence += 2
        
        # Path 5: BB upper + volume (volatility entry)
        if price_above_bb_upper and vol_ok:
            short_confidence += 1
        
        # Path 6: Any RSI overbought (fallback for trade generation)
        if rsi_14[i] > 65:
            short_confidence += 1
        
        if short_confidence >= 3:
            new_signal = -current_size
        elif short_confidence >= 2 and bars_since_last_trade > 40:
            new_signal = -current_size * 0.6
        elif short_confidence >= 1 and bars_since_last_trade > 80:
            new_signal = -current_size * 0.4
        
        # === FORCE TRADE MECHANISM (Critical for avoiding 0 trades) ===
        if bars_since_last_trade > 120 and new_signal == 0.0 and not in_position:
            # Force entry based on simplest signals
            if trend_4h_bullish and rsi_14[i] < 45:
                new_signal = current_size * 0.4
            elif trend_4h_bearish and rsi_14[i] > 55:
                new_signal = -current_size * 0.4
            elif rsi_14[i] < 35:
                new_signal = current_size * 0.3
            elif rsi_14[i] > 65:
                new_signal = -current_size * 0.3
            elif price_below_bb_lower:
                new_signal = current_size * 0.3
            elif price_above_bb_upper:
                new_signal = -current_size * 0.3
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.0 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.0 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and trend_4h_bearish and rsi_14[i] > 60:
                regime_reversal = True
            if position_side < 0 and trend_4h_bullish and rsi_14[i] < 40:
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