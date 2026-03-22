#!/usr/bin/env python3
"""
Experiment #045: 1h HTF Trend + RSI Pullback + Volume/Session Filter

Hypothesis: Current best (4h HMA+RSI, Sharpe=0.028) works but needs adaptation for 1h.
Key insight: 1h generates more signals → need STRICTER filters to avoid fee drag.

Strategy components:
1. 1d HMA(21) for major trend bias (slowest, most reliable)
2. 4h HMA(21) for intermediate trend confirmation
3. 1h RSI(14) pullback entries within HTF trend
4. Volume filter: volume > 0.8x 20-bar average (confirm participation)
5. Session filter: 8-20 UTC only (liquidity hours, avoid Asian chop)
6. ATR(14) trailing stoploss at 2.5x

Why this should beat Sharpe=0.028:
- Triple HTF confirmation (1d + 4h) reduces false signals
- Session filter eliminates low-liquidity whipsaws
- Volume confirmation ensures real moves, not fakeouts
- 1h entry timing captures better R:R than 4h entries
- Discrete sizing (0.25) controls drawdown in 2022-style crashes

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h and 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 base (smaller for 1h due to higher trade frequency)
Stoploss: 2.5 * ATR(14) trailing
Trade target: 40-60/year (strict filters to avoid >100/year fee drag)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_htf_trend_rsi_session_vol_v1"
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
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_volume_avg(volume, period=20):
    """Calculate rolling average volume."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_avg

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
    ts_seconds = open_time / 1000
    utc_hour = pd.to_datetime(ts_seconds, unit='s').hour
    return utc_hour

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    vol_avg_20 = calculate_volume_avg(volume, 20)
    
    # Extract UTC hours for session filter
    utc_hours = np.array([get_utc_hour(ot) for ot in open_time])
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    # Smaller size for 1h due to higher trade frequency
    BASE_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -100
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(vol_avg_20[i]):
            continue
        
        # === 1D MAJOR TREND BIAS ===
        trend_1d_bullish = close[i] > hma_1d_21_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === 4H INTERMEDIATE TREND ===
        trend_4h_bullish = close[i] > hma_4h_21_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_21_aligned[i]
        
        # === TREND CONFLUENCE (both 1d and 4h agree) ===
        strong_bullish = trend_1d_bullish and trend_4h_bullish
        strong_bearish = trend_1d_bearish and trend_4h_bearish
        
        # === RSI PULLBACK SIGNALS ===
        # Long: RSI pulled back to 35-45 in bullish trend
        # Short: RSI rallied to 55-65 in bearish trend
        rsi_long_pullback = 35 <= rsi_14[i] <= 50
        rsi_short_pullback = 50 <= rsi_14[i] <= 65
        
        # === VOLUME FILTER ===
        # Volume must be >= 0.8x average (confirm participation)
        volume_ok = volume[i] >= 0.8 * vol_avg_20[i]
        
        # === SESSION FILTER (8-20 UTC) ===
        # Only trade during high-liquidity hours
        session_ok = 8 <= utc_hours[i] <= 20
        
        # === VOLATILITY-ADJUSTED POSITION SIZING ===
        if i > 100:
            atr_median = np.nanmedian(atr_14[max(0, i-100):i])
            atr_ratio = atr_14[i] / atr_median if atr_median > 0 else 1.0
            vol_adjustment = np.clip(1.0 / atr_ratio, 0.7, 1.3)
        else:
            vol_adjustment = 1.0
        
        current_size = BASE_SIZE * vol_adjustment
        current_size = np.clip(current_size, 0.20, 0.30)
        # Round to discrete levels
        if current_size > 0.27:
            current_size = 0.30
        elif current_size > 0.17:
            current_size = 0.20
        else:
            current_size = 0.15
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES: Strong bullish trend + RSI pullback + volume + session
        if strong_bullish and rsi_long_pullback and volume_ok and session_ok:
            new_signal = current_size
        
        # SHORT ENTRIES: Strong bearish trend + RSI pullback + volume + session
        elif strong_bearish and rsi_short_pullback and volume_ok and session_ok:
            new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 72 bars (~3 days on 1h), allow weaker entry
        if bars_since_last_trade > 72 and new_signal == 0.0 and not in_position:
            if strong_bullish and 30 <= rsi_14[i] <= 55:
                new_signal = current_size * 0.5
            elif strong_bearish and 45 <= rsi_14[i] <= 70:
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
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and strong_bearish:
                trend_reversal = True
            if position_side < 0 and strong_bullish:
                trend_reversal = True
        
        # Apply stoploss or trend reversal
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