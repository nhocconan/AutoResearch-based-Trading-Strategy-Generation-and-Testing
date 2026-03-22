#!/usr/bin/env python3
"""
Experiment #175: 1h Primary + 4h/1d HTF — Simplified Regime + RSI Pullback

Hypothesis: Recent 1h strategies failed (Sharpe=0.000) because they had TOO MANY
confluence filters that never all aligned. This strategy uses SIMPLER logic:

1. 4h HMA(21) slope for trend bias (proven in best strategies)
2. 1h RSI(14) for entry timing (simpler than Connors RSI, more reliable)
3. Bollinger Bands(20, 2.0) for mean reversion confirmation
4. Session filter: 8-20 UTC only (naturally limits trades to ~40-60/year)
5. Volume filter: volume > 0.8x 20-bar average
6. ATR(14) trailing stop: 2.0x ATR

Why this should work:
- Simpler conditions = more trades (avoiding 0-trade failure)
- Session filter naturally limits frequency without complex logic
- 4h trend + 1h pullback = proven pattern from best strategies
- RSI(14) extremes happen regularly (unlike Connors RSI <15)
- Position size 0.25 (conservative for 1h TF)

Timeframe: 1h (REQUIRED)
HTF: 4h via mtf_data.get_htf_data() — called ONCE before loop
Target trades: 40-80/year per symbol (session filter ensures this)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_rsi_hma4h_session_v1"
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

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    return (open_time // (1000 * 60 * 60)) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
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
    rsi_14 = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    
    # Volume average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Extract UTC hours for session filter
    utc_hours = np.array([get_utc_hour(ot) for ot in open_time])
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.25  # Conservative for 1h TF
    
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
        
        if np.isnan(vol_avg_20[i]) or vol_avg_20[i] == 0:
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        current_hour = utc_hours[i]
        in_session = 8 <= current_hour <= 20
        
        # === 4H TREND BIAS ===
        trend_4h_bullish = hma_4h_slope_aligned[i] > 0.5
        trend_4h_bearish = hma_4h_slope_aligned[i] < -0.5
        trend_4h_neutral = not trend_4h_bullish and not trend_4h_bearish
        
        price_above_4h_hma = close[i] > hma_4h_21_aligned[i]
        price_below_4h_hma = close[i] < hma_4h_21_aligned[i]
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * vol_avg_20[i]
        
        # === RSI CONDITIONS ===
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        rsi_extreme_low = rsi_14[i] < 25
        rsi_extreme_high = rsi_14[i] > 75
        
        # === BOLLINGER BAND POSITION ===
        price_below_bb_lower = close[i] < bb_lower[i]
        price_above_bb_upper = close[i] > bb_upper[i]
        
        # === ENTRY LOGIC (OR paths for more trades) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Multiple paths (any one can trigger)
        long_entry = False
        
        # Path 1: Bullish 4h trend + RSI pullback (primary)
        if in_session and trend_4h_bullish and rsi_oversold and volume_ok:
            long_entry = True
        
        # Path 2: Neutral 4h + BB lower + RSI extreme (mean revert)
        if in_session and trend_4h_neutral and price_below_bb_lower and rsi_extreme_low:
            long_entry = True
        
        # Path 3: Price above 4h HMA + RSI very low (deep pullback)
        if in_session and price_above_4h_hma and rsi_14[i] < 30:
            long_entry = True
        
        # Path 4: Simple oversold in session (fallback for trade frequency)
        if in_session and rsi_extreme_low and bars_since_last_trade > 48:
            long_entry = True
        
        if long_entry:
            new_signal = BASE_SIZE
        
        # SHORT ENTRIES
        short_entry = False
        
        # Path 1: Bearish 4h trend + RSI rally (primary)
        if in_session and trend_4h_bearish and rsi_overbought and volume_ok:
            short_entry = True
        
        # Path 2: Neutral 4h + BB upper + RSI extreme (mean revert)
        if in_session and trend_4h_neutral and price_above_bb_upper and rsi_extreme_high:
            short_entry = True
        
        # Path 3: Price below 4h HMA + RSI very high (rally in bear)
        if in_session and price_below_4h_hma and rsi_14[i] > 70:
            short_entry = True
        
        # Path 4: Simple overbought in session (fallback)
        if in_session and rsi_extreme_high and bars_since_last_trade > 48:
            short_entry = True
        
        if short_entry:
            new_signal = -BASE_SIZE
        
        # === FREQUENCY SAFEGUARD (ensure minimum trades) ===
        if bars_since_last_trade > 120 and new_signal == 0.0 and not in_position and in_session:
            # Force entry if no trades for 5 days (120 hours)
            if trend_4h_bullish and rsi_14[i] < 40:
                new_signal = BASE_SIZE * 0.5
            elif trend_4h_bearish and rsi_14[i] > 60:
                new_signal = -BASE_SIZE * 0.5
            elif rsi_14[i] < 30:
                new_signal = BASE_SIZE * 0.4
            elif rsi_14[i] > 70:
                new_signal = -BASE_SIZE * 0.4
        
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