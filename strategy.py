#!/usr/bin/env python3
"""
Experiment #080: 1h Primary + 4h/12h HTF — Simplified Regime + RSI Pullback

Hypothesis: Previous 1h/30m strategies (#075, #078) failed with 0 trades because:
1. Connors RSI calculation had NaN/edge cases
2. Too many AND conditions (session + volume + regime + indicator)
3. Entry thresholds too strict for actual market conditions

This strategy SIMPLIFIES entry logic while keeping HTF trend filter:
1. 4h HMA(21) slope = major trend bias (call ONCE before loop)
2. 12h HMA(21) = secondary trend confirmation (call ONCE before loop)
3. 1h RSI(14) for entry timing (oversold <40, overbought >60) - SOFTER thresholds
4. 1h Choppiness(14) for regime (range >55, trend <45) - transitional allowed
5. Volume filter: >0.7x 20-bar avg (softer than 0.8x)
6. Session filter: 8-20 UTC BUT allow 30% of trades outside for frequency
7. ATR(14) stoploss at 2.5x trailing
8. Position size: 0.25 discrete (balanced for 1h TF)

Key changes from failed #075/#078:
- RSI instead of CRSI (simpler, more reliable)
- Softer thresholds (RSI 40/60 vs CRSI 20/80)
- Volume filter 0.7x not 0.8x
- Session filter has 30% bypass for trade frequency
- Frequency safeguard: force entry if no trades for 200 bars (~8 days)

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h + 12h via mtf_data.get_htf_data() — called ONCE before loop
Target trades: 40-80/year per symbol (1h with HTF filter)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_simp_regime_rsi_4h12h_v1"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    """
    atr_values = calculate_atr(high, low, close, period)
    
    atr_sum = pd.Series(atr_values).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    price_range = np.where(price_range == 0, 1e-10, price_range)
    
    chop = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    
    return chop

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
    """Calculate HMA slope as percentage change over lookback."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        if hma_values[i - lookback] != 0 and not np.isnan(hma_values[i - lookback]):
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback] * 100
    return slope

def get_session_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
    hour = pd.to_datetime(open_time, unit='ms').hour
    return hour

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate HTF indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_slope = calculate_hma_slope(hma_4h_21, 5)
    
    hma_12h_21 = calculate_hma(df_12h['close'].values, 21)
    hma_12h_slope = calculate_hma_slope(hma_12h_21, 5)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_slope_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_slope)
    hma_12h_slope_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_slope)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    
    # Volume average (20-bar)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -100
    trades_outside_session = 0
    max_trades_outside_session = None  # Will set after first 500 bars
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_slope_aligned[i]) or np.isnan(hma_12h_slope_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]):
            continue
        
        if np.isnan(vol_avg_20[i]) or vol_avg_20[i] == 0:
            continue
        
        # === HTF TREND BIAS (4h + 12h) ===
        # Both 4h and 12h slope > 0 = strong bullish
        # Both 4h and 12h slope < 0 = strong bearish
        # Mixed = neutral (reduce position size)
        trend_4h_bullish = hma_4h_slope_aligned[i] > 0.3
        trend_4h_bearish = hma_4h_slope_aligned[i] < -0.3
        trend_12h_bullish = hma_12h_slope_aligned[i] > 0.3
        trend_12h_bearish = hma_12h_slope_aligned[i] < -0.3
        
        strong_bullish = trend_4h_bullish and trend_12h_bullish
        strong_bearish = trend_4h_bearish and trend_12h_bearish
        neutral_trend = not strong_bullish and not strong_bearish
        
        # === CHOPPINESS REGIME ===
        # CHOP > 55 = range market (mean revert)
        # CHOP < 45 = trend market (trend follow)
        # Between = transitional (allow both)
        is_range = chop_14[i] > 55
        is_trend = chop_14[i] < 45
        
        # === RSI ENTRY SIGNALS ===
        # Range market: extreme RSI for mean reversion
        # Trend market: moderate RSI for pullback entries
        rsi_oversold = rsi_14[i] < 40  # Softer than 30
        rsi_overbought = rsi_14[i] > 60  # Softer than 70
        rsi_neutral_low = rsi_14[i] < 50
        rsi_neutral_high = rsi_14[i] > 50
        
        # === VOLUME FILTER ===
        vol_ok = volume[i] > 0.7 * vol_avg_20[i]
        
        # === SESSION FILTER ===
        hour = get_session_hour(open_time[i])
        in_session = 8 <= hour <= 20
        
        # Allow 30% of trades outside session for frequency
        allow_outside_session = (max_trades_outside_session is None) or \
                                (trades_outside_session < max_trades_outside_session * 0.5)
        
        session_ok = in_session or allow_outside_session
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        if neutral_trend:
            current_size = BASE_SIZE * 0.6
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # Set max_trades_outside_session after 500 bars
        if max_trades_outside_session is None and i > 500 and last_trade_bar > 0:
            # Estimate: allow ~30% of trades outside session
            total_trades_est = max(1, (i - last_trade_bar) // 50)
            max_trades_outside_session = max(5, total_trades_est)
        
        # LONG ENTRIES
        long_condition = False
        
        if is_range:
            # Mean reversion: buy oversold in range
            if rsi_oversold and (strong_bullish or neutral_trend):
                long_condition = True
        elif is_trend:
            # Trend follow: buy pullback in uptrend
            if strong_bullish and rsi_neutral_low:
                long_condition = True
        else:
            # Transitional: any bullish signal
            if (trend_4h_bullish or trend_12h_bullish) and rsi_oversold:
                long_condition = True
        
        if long_condition and vol_ok and session_ok:
            new_signal = current_size
            if not in_session:
                trades_outside_session += 1
        
        # SHORT ENTRIES
        short_condition = False
        
        if is_range:
            # Mean reversion: sell overbought in range
            if rsi_overbought and (strong_bearish or neutral_trend):
                short_condition = True
        elif is_trend:
            # Trend follow: sell pullback in downtrend
            if strong_bearish and rsi_neutral_high:
                short_condition = True
        else:
            # Transitional: any bearish signal
            if (trend_4h_bearish or trend_12h_bearish) and rsi_overbought:
                short_condition = True
        
        if short_condition and vol_ok and session_ok and new_signal == 0.0:
            new_signal = -current_size
            if not in_session:
                trades_outside_session += 1
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 200 bars (~8 days on 1h), allow weaker entry
        if bars_since_last_trade > 200 and new_signal == 0.0 and not in_position:
            if trend_4h_bullish and rsi_14[i] < 45:
                new_signal = current_size * 0.5
            elif trend_4h_bearish and rsi_14[i] > 55:
                new_signal = -current_size * 0.5
        
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
        
        # Apply stoploss
        if stoploss_triggered:
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
                # Position flip
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