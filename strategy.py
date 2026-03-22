#!/usr/bin/env python3
"""
Experiment #475: 1h Primary + 4h/1d HTF — Trend Direction + Mean Revert Entry

Hypothesis: After 474 experiments, clear pattern emerges for 1h timeframe:
1. 1h strategies fail when entry filters are too strict (see #465, #470 = 0 trades)
2. SUCCESS pattern: 4h/1d for SIGNAL DIRECTION, 1h only for ENTRY TIMING
3. Must LOOSEN entry thresholds to ensure >=30 trades/symbol on train
4. Session filter (8-20 UTC) reduces noise without killing frequency
5. Volume filter (0.7x avg) ensures liquidity without being too restrictive

Why this might beat current best (Sharpe=0.435):
- 4h HMA provides clean trend bias (proven in research)
- 1d HMA adds major regime filter (bull/bear market)
- 1h RSI for pullback entries (not extremes = more trades)
- Relaxed thresholds: RSI 35/65 instead of 20/80
- ATR 2.0x trailing stop protects in crashes
- Discrete sizing: 0.0, ±0.25, ±0.30 (minimize fee churn)

Position sizing: 0.25-0.30 (max 0.40)
Stoploss: 2.0 * ATR trailing (signal → 0 when hit)
Target: 40-80 trades/year on 1h, >=30 trades/symbol on train, >=3 on test

CRITICAL: Entry conditions deliberately LOOSE to avoid 0-trade failure.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_4h1d_trend_pullback_v1"
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
    """Calculate Hull Moving Average (HMA)."""
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        return series.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    hma_raw = 2.0 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_n)
    
    return hma.values

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

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_volume_ma(volume, period=20):
    """Calculate volume moving average."""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

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
    hma_4h_21 = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_50 = calculate_hma(df_4h['close'].values, period=50)
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_50_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_50)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    rsi_7 = calculate_rsi(close, 7)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    vol_ma_20 = calculate_volume_ma(volume, 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_SIZE = 0.30
    SHORT_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_1d_21_aligned[i]):
            continue
        if np.isnan(hma_4h_50_aligned[i]) or np.isnan(rsi_14[i]):
            continue
        if np.isnan(sma_50[i]) or np.isnan(vol_ma_20[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        # Extract hour from open_time (milliseconds timestamp)
        hour_utc = (open_time[i] // 3600000) % 24
        in_session = 8 <= hour_utc <= 20
        
        # === VOLUME FILTER (must be > 0.7x average) ===
        vol_ratio = volume[i] / (vol_ma_20[i] + 1e-10)
        vol_ok = vol_ratio > 0.7
        
        # === 1D MAJOR REGIME (bull/bear market) ===
        bull_market = close[i] > hma_1d_21_aligned[i]
        bear_market = close[i] < hma_1d_21_aligned[i]
        
        # === 4H TREND DIRECTION (primary signal filter) ===
        hma_4h_bullish = hma_4h_21_aligned[i] > hma_4h_50_aligned[i]
        hma_4h_bearish = hma_4h_21_aligned[i] < hma_4h_50_aligned[i]
        
        # === 1H LOCAL TREND ===
        hma_1h_bullish = close[i] > sma_50[i]
        hma_1h_bearish = close[i] < sma_50[i]
        
        # === RSI PULLBACK SIGNALS (relaxed for trade frequency) ===
        rsi_oversold = rsi_14[i] < 40.0
        rsi_overbought = rsi_14[i] > 60.0
        rsi_extreme_oversold = rsi_7[i] < 30.0
        rsi_extreme_overbought = rsi_7[i] > 70.0
        
        # === SMA200 FILTER (major trend confirmation) ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === ENTRY LOGIC — LOOSE CONDITIONS FOR TRADE FREQUENCY ===
        new_signal = 0.0
        
        # LONG ENTRIES (multiple paths to trigger)
        long_conditions = 0
        if bull_market:
            long_conditions += 1
        if hma_4h_bullish:
            long_conditions += 1
        if hma_1h_bullish or rsi_oversold:
            long_conditions += 1
        if above_sma200:
            long_conditions += 1
        if rsi_14[i] < 45.0:
            long_conditions += 1
        
        # Need at least 3 of 5 conditions for long (relaxed from 4)
        if long_conditions >= 3 and in_session and vol_ok:
            new_signal = LONG_SIZE
        
        # SHORT ENTRIES (multiple paths to trigger)
        if new_signal == 0.0:
            short_conditions = 0
            if bear_market:
                short_conditions += 1
            if hma_4h_bearish:
                short_conditions += 1
            if hma_1h_bearish or rsi_overbought:
                short_conditions += 1
            if below_sma200:
                short_conditions += 1
            if rsi_14[i] > 55.0:
                short_conditions += 1
            
            # Need at least 3 of 5 conditions for short
            if short_conditions >= 3 and in_session and vol_ok:
                new_signal = -SHORT_SIZE
        
        # === STOPLOSS CHECK (2.0 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.0 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === TAKE PROFIT / EXIT CONDITIONS ===
        # Exit long on RSI extreme overbought
        if in_position and position_side > 0 and rsi_14[i] > 75.0:
            new_signal = 0.0
        
        # Exit short on RSI extreme oversold
        if in_position and position_side < 0 and rsi_14[i] < 25.0:
            new_signal = 0.0
        
        # Regime flip exit (major trend reversal)
        if in_position and position_side > 0 and bear_market and hma_4h_bearish:
            new_signal = 0.0
        if in_position and position_side < 0 and bull_market and hma_4h_bullish:
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