#!/usr/bin/env python3
"""
Experiment #635: 1h Primary + 4h/1d HTF — HMA Trend + RSI Pullback + Volatility Filter

Hypothesis: Building on lessons from 561 failed strategies, this approach simplifies
the entry logic while maintaining multi-timeframe confluence. Recent failures show
that Choppiness+CRSI combinations produce 0 trades (#628, #632). Instead, we use:

1. 1d HMA slope for major trend direction (slow, reliable filter)
2. 4h RSI for pullback detection within the trend
3. 1h price action for entry timing (break of recent high/low)
4. ATR volatility filter to skip low-vol chop periods
5. Session filter (8-20 UTC) to reduce trade frequency

Key insights from failures:
- Over-filtering = 0 trades (see #628, #632 with Sharpe=0.000)
- Lower TF needs HTF direction filter to avoid whipsaws
- RSI pullback in trend direction has proven win rate
- Volume/session filters critical for 1h to hit 30-60 trades/year target

Why this might beat Sharpe=0.520:
- Simpler entry logic = more reliable signals
- 1d trend filter keeps us on right side of major moves
- 4h RSI pullback entries have high win rate in trending markets
- ATR volatility filter skips dead chop periods
- Session filter reduces trades to target range (30-60/year)

Position sizing: 0.25 discrete (conservative for 1h TF)
Target: 30-60 trades/year on 1h (per Rule 10)
Stoploss: 2.5*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_vol_4h1d_v1"
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
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    Faster response than EMA with less lag.
    """
    close_s = pd.Series(close)
    
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

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
    
    # Calculate 1d HMA for major trend direction
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h RSI for pullback detection
    rsi_4h = calculate_rsi(df_4h['close'].values, period=14)
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # Calculate 1h indicators
    hma_1h = calculate_hma(close, period=21)
    hma_1h_fast = calculate_hma(close, period=9)
    atr_14 = calculate_atr(high, low, close, 14)
    atr_30 = calculate_atr(high, low, close, 30)
    rsi_14 = calculate_rsi(close, 14)
    sma_200 = calculate_sma(close, 200)
    
    # Volume moving average
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(hma_1h[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(rsi_4h_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(sma_200[i]) or np.isnan(vol_sma_20[i]):
            continue
        if atr_14[i] == 0 or atr_30[i] == 0:
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        # Extract hour from open_time (milliseconds timestamp)
        hour_utc = (open_time[i] // 3600000) % 24
        in_session = 8 <= hour_utc <= 20
        
        # === VOLUME FILTER ===
        vol_ok = volume[i] > 0.8 * vol_sma_20[i]
        
        # === VOLATILITY FILTER (skip dead chop) ===
        # ATR ratio > 0.7 means we're not in extremely low vol
        vol_ratio = atr_14[i] / (atr_30[i] + 1e-10)
        vol_filter = vol_ratio > 0.6
        
        # === 1D TREND BIAS (HMA slope over 5 bars) ===
        hma_1d_slope_bull = hma_1d_aligned[i] > hma_1d_aligned[i-5] if i >= 5 else False
        hma_1d_slope_bear = hma_1d_aligned[i] < hma_1d_aligned[i-5] if i >= 5 else False
        
        # Price relative to 1d HMA and 200 SMA
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        price_above_sma_200 = close[i] > sma_200[i]
        price_below_sma_200 = close[i] < sma_200[i]
        
        # === 4H RSI PULLBACK ZONES ===
        # In uptrend: look for RSI pullback to 35-50
        # In downtrend: look for RSI bounce to 50-65
        rsi_pullback_long = 35.0 <= rsi_4h_aligned[i] <= 55.0
        rsi_pullback_short = 45.0 <= rsi_4h_aligned[i] <= 65.0
        rsi_oversold = rsi_4h_aligned[i] < 40.0
        rsi_overbought = rsi_4h_aligned[i] > 60.0
        
        # === 1H HMA FAST/SLOW CROSSOVER ===
        hma_cross_bull = hma_1h_fast[i] > hma_1h[i]
        hma_cross_bear = hma_1h_fast[i] < hma_1h[i]
        
        # === 1H HMA SLOPE (3 bars) ===
        hma_1h_slope_bull = hma_1h[i] > hma_1h[i-3] if i >= 3 else False
        hma_1h_slope_bear = hma_1h[i] < hma_1h[i-3] if i >= 3 else False
        
        # === PRICE BREAKOUT (recent high/low) ===
        recent_high = np.max(high[i-10:i]) if i >= 10 else high[i]
        recent_low = np.min(low[i-10:i]) if i >= 10 else low[i]
        breakout_up = close[i] > recent_high * 0.998
        breakout_down = close[i] < recent_low * 1.002
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY: 1d bull trend + 4h RSI pullback + 1h confirmation ---
        # Must pass: session, volume, volatility filters
        if in_session and vol_ok and vol_filter:
            if hma_1d_slope_bull and price_above_hma_1d and price_above_sma_200:
                if rsi_oversold or rsi_pullback_long:
                    if hma_cross_bull and hma_1h_slope_bull:
                        if breakout_up:
                            new_signal = POSITION_SIZE
        
        # --- SHORT ENTRY: 1d bear trend + 4h RSI bounce + 1h confirmation ---
        # Must pass: session, volume, volatility filters
        if in_session and vol_ok and vol_filter:
            if hma_1d_slope_bear and price_below_hma_1d and price_below_sma_200:
                if rsi_overbought or rsi_pullback_short:
                    if hma_cross_bear and hma_1h_slope_bear:
                        if breakout_down:
                            new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON TREND FLIP ===
        if in_position and position_side > 0:
            if hma_1d_slope_bear and price_below_hma_1d:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if hma_1d_slope_bull and price_above_hma_1d:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals