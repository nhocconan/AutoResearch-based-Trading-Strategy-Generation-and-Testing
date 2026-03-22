#!/usr/bin/env python3
"""
Experiment #085: 1h Primary + 4h/1d HTF — Regime-Adaptive RSI with Session/Volume Filters

Hypothesis: Previous 1h strategies (#075, #080) failed with Sharpe=0.000 because entry
conditions were TOO STRICT (multiple conflicting filters never aligned). This strategy:
1. Uses LOOSER entry thresholds (RSI 30/70 instead of 20/80) to ensure trades
2. 4h HMA for TREND DIRECTION (primary signal driver)
3. 1h Choppiness for REGIME (range=tighter stops, trend=larger size)
4. 1h RSI for ENTRY TIMING (pullback within HTF trend)
5. Session filter (8-20 UTC) for liquidity
6. Volume filter (>0.7x avg) for confirmation
7. Frequency safeguard: if no trade in 150 bars, allow weaker entry

Why this should work:
- 4h trend direction prevents counter-trend trades (major filter)
- Looser RSI thresholds ensure we generate trades (>10 train, >3 test)
- Session/volume filters reduce false signals during low liquidity
- Regime-adaptive sizing: larger in trends, smaller in ranges
- 1h timeframe with HTF direction = HTF trade frequency with 1h precision

Timeframe: 1h (REQUIRED)
HTF: 4h via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25 discrete (smaller for 1h due to more potential trades)
Stoploss: 2.0 * ATR(14) trailing
Target trades: 40-80/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_rsi_session_vol_4h_v1"
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
    CHOP > 61.8 = range market (mean revert)
    CHOP < 38.2 = trend market (trend follow)
    """
    atr_values = calculate_atr(high, low, close, period)
    
    # Rolling sum of ATR
    atr_sum = pd.Series(atr_values).rolling(window=period, min_periods=period).sum().values
    
    # Highest High and Lowest Low over period
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Choppiness calculation
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
    """Calculate HMA slope over lookback period."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        if hma_values[i - lookback] != 0:
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback] * 100
    return slope

def extract_hour_from_open_time(open_time_array):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
    hours = ((open_time_array // 1000) // 3600) % 24
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
    
    # Calculate HTF indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_slope = calculate_hma_slope(hma_4h_21, 5)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_slope_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_slope)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    
    # Volume moving average for filter
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session hours (8-20 UTC = high liquidity)
    session_hours = extract_hour_from_open_time(open_time)
    in_session = (session_hours >= 8) & (session_hours <= 20)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    # Smaller size for 1h due to more potential trades
    BASE_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -100
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_4h_slope_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(rsi_14[i]):
            continue
        
        if np.isnan(vol_sma_20[i]) or vol_sma_20[i] == 0:
            continue
        
        # === 4H TREND DIRECTION (PRIMARY SIGNAL) ===
        # HMA slope > 0.3 = bullish trend (prefer longs)
        # HMA slope < -0.3 = bearish trend (prefer shorts)
        # Use looser threshold to ensure we get signals
        trend_4h_bullish = hma_4h_slope_aligned[i] > 0.2
        trend_4h_bearish = hma_4h_slope_aligned[i] < -0.2
        
        # Price vs 4h HMA for additional confirmation
        price_above_4h_hma = close[i] > hma_4h_21_aligned[i]
        price_below_4h_hma = close[i] < hma_4h_21_aligned[i]
        
        # === CHOPPINESS REGIME DETECTION ===
        # CHOP > 55 = range market (mean revert, smaller size)
        # CHOP < 45 = trend market (trend follow, larger size)
        # Use looser thresholds to ensure regime detection works
        is_range_market = chop_14[i] > 50
        is_trend_market = chop_14[i] < 50
        
        # === RSI ENTRY SIGNALS (LOOSER THRESHOLDS) ===
        # Use 30/70 instead of 20/80 to ensure we get trades
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        rsi_neutral_low = rsi_14[i] < 45
        rsi_neutral_high = rsi_14[i] > 55
        
        # === VOLUME CONFIRMATION ===
        # Volume > 0.7x average (looser than 0.8x)
        volume_ok = volume[i] > 0.7 * vol_sma_20[i]
        
        # === SESSION FILTER ===
        # Only trade during high liquidity hours (8-20 UTC)
        # But allow breakout trades outside session if strong signal
        session_ok = in_session[i]
        
        # === POSITION SIZING BY REGIME ===
        if is_trend_market:
            current_size = BASE_SIZE  # Full size in trends
        else:
            current_size = BASE_SIZE * 0.7  # Reduced size in ranges
        
        # === ENTRY LOGIC (LOOSE ENOUGH TO GENERATE TRADES) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        # Primary: 4h bullish + RSI pullback + volume
        if trend_4h_bullish and rsi_oversold and volume_ok:
            new_signal = current_size
        # Secondary: price above 4h HMA + RSI neutral low (looser)
        elif price_above_4h_hma and rsi_neutral_low and volume_ok:
            new_signal = current_size * 0.7
        # Tertiary: 4h bullish + RSI < 50 (very loose, for trade generation)
        elif trend_4h_bullish and rsi_14[i] < 50:
            new_signal = current_size * 0.5
        
        # SHORT ENTRIES
        # Primary: 4h bearish + RSI pullback + volume
        if trend_4h_bearish and rsi_overbought and volume_ok:
            new_signal = -current_size
        # Secondary: price below 4h HMA + RSI neutral high (looser)
        elif price_below_4h_hma and rsi_neutral_high and volume_ok:
            new_signal = -current_size * 0.7
        # Tertiary: 4h bearish + RSI > 50 (very loose, for trade generation)
        elif trend_4h_bearish and rsi_14[i] > 50:
            new_signal = -current_size * 0.5
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 150 bars (~6 days on 1h), allow much weaker entry
        # This ensures we generate minimum trades even in quiet markets
        if bars_since_last_trade > 150 and new_signal == 0.0 and not in_position:
            if trend_4h_bullish and rsi_14[i] < 55:
                new_signal = current_size * 0.4
            elif trend_4h_bearish and rsi_14[i] > 45:
                new_signal = -current_size * 0.4
        
        # === SESSION OVERRIDE FOR STRONG SIGNALS ===
        # If signal is strong (primary entry), ignore session filter
        # If signal is weak (tertiary), require session
        if new_signal != 0.0:
            signal_strength = abs(new_signal) / current_size
            if signal_strength < 0.6 and not session_ok:
                # Weak signal outside session = skip
                new_signal = 0.0
        
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
            # If same side, keep position (no update needed)
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