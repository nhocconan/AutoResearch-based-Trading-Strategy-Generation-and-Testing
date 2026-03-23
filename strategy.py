#!/usr/bin/env python3
"""
Experiment #1085: 1h Primary + 4h/1d HTF — Dual HTF Trend + RSI Pullback + Session/Volume Filter

Hypothesis: After 785+ failed experiments, 1h strategies fail due to:
1. Too many trades → fee drag kills profit (>200/year = death)
2. No HTF alignment → counter-trend trades in 2022 crash
3. No session filter → entries during low-liquidity Asian overnight

Solution:
1. DUAL HTF FILTER: Both 4h AND 1d HMA21 must agree on trend direction
   - This cuts trades by ~60% but dramatically improves quality
2. 1h RSI pullback entries (40-55 long, 45-60 short) — NOT extremes
   - Generates entries within trend, not at reversals
3. SESSION FILTER: Only 8-20 UTC (London/NY overlap = high liquidity)
   - Avoids Asian overnight whipsaws
4. VOLUME FILTER: Volume > 0.8x 20-period average
   - Confirms institutional participation
5. ATR(14) trailing stop 2.5x — proper risk management
6. Position size: 0.20-0.30 discrete (smaller for 1h vs 4h)

Why this should beat Sharpe=0.612:
- Dual HTF (4h+1d) prevents counter-trend disasters in 2022
- Session filter alone cuts 40% of low-quality trades
- Volume filter avoids fake breakouts
- RSI pullback generates 40-70 trades/year (sweet spot for 1h)
- Simpler than CRSI/Choppiness = more robust

Timeframe: 1h (primary)
HTF: 4h + 1d — loaded ONCE before loop using mtf_data helper
Position Size: 0.20-0.30 discrete levels
Stoploss: 2.5x ATR trailing
Target trades: 40-80/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_dual_htf_hma_rsi_session_vol_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(series, period):
    """
    Hull Moving Average — faster and smoother than EMA.
    
    Formula:
    1. WMA(period/2) * 2
    2. WMA(period) * 1
    3. Diff = (1) - (2)
    4. HMA = WMA(sqrt(period)) of Diff
    """
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_rsi(close, period=14):
    """
    Relative Strength Index — momentum oscillator.
    
    Formula:
    1. Calculate gains and losses
    2. EMA of gains and losses over period
    3. RSI = 100 - (100 / (1 + RS)) where RS = avg_gain / avg_loss
    """
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    diff = np.diff(close)
    gain = np.where(diff > 0, diff, 0.0)
    loss = np.where(diff < 0, -diff, 0.0)
    
    # Pad first element
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 1e-10
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100.0 - (100.0 / (1.0 + rs[mask]))
    rsi[~mask] = 50.0
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement and stoploss."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_sma(volume, period=20):
    """Simple moving average of volume for volume filter."""
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
    ts_seconds = open_time / 1000.0
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
    
    # Calculate and align 4h HMA21 for intermediate trend
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA21 for macro trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (1h) indicators
    hma_21 = calculate_hma(close, 21)
    hma_48 = calculate_hma(close, 48)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    vol_sma = calculate_volume_sma(volume, period=20)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_21[i]) or np.isnan(hma_48[i]):
            continue
        if np.isnan(rsi[i]) or np.isnan(atr[i]) or np.isnan(vol_sma[i]):
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if atr[i] <= 1e-10 or vol_sma[i] <= 1e-10:
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        utc_hour = get_utc_hour(open_time[i])
        in_session = 8 <= utc_hour <= 20
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * vol_sma[i]
        
        # === MACRO TREND (1d HMA21) — SUPER MACRO ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === INTERMEDIATE TREND (4h HMA21) ===
        inter_bull = close[i] > hma_4h_aligned[i]
        inter_bear = close[i] < hma_4h_aligned[i]
        
        # === PRIMARY TREND (1h HMA crossover) ===
        hma_bull = hma_21[i] > hma_48[i]
        hma_bear = hma_21[i] < hma_48[i]
        
        # === RSI PULLBACK SIGNALS ===
        # Long: RSI pulled back to 40-55 in uptrend
        rsi_pullback_long = 40.0 <= rsi[i] <= 55.0
        # Short: RSI rallied to 45-60 in downtrend
        rsi_pullback_short = 45.0 <= rsi[i] <= 60.0
        
        # === VOLATILITY CHECK ===
        vol_spike = atr[i] > 2.0 * np.nanmedian(atr[max(0, i-100):i]) if i > 100 else False
        current_size = REDUCED_SIZE if vol_spike else BASE_SIZE
        
        desired_signal = 0.0
        
        # === LONG ENTRY ===
        # ALL conditions must align: session + volume + 1d bull + 4h bull + 1h HMA bull + RSI pullback
        if in_session and volume_ok and macro_bull and inter_bull and hma_bull and rsi_pullback_long:
            desired_signal = current_size
        
        # === SHORT ENTRY ===
        # ALL conditions must align: session + volume + 1d bear + 4h bear + 1h HMA bear + RSI pullback
        elif in_session and volume_ok and macro_bear and inter_bear and hma_bear and rsi_pullback_short:
            desired_signal = -current_size
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if trend intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if 4h still bullish and RSI not overbought
                if inter_bull and rsi[i] < 70.0:
                    desired_signal = current_size
            elif position_side < 0:
                # Hold short if 4h still bearish and RSI not oversold
                if inter_bear and rsi[i] > 30.0:
                    desired_signal = -current_size
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if 4h crosses bearish or 1d reverses
            if inter_bear and rsi[i] > 65.0:
                desired_signal = 0.0
            if macro_bear:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 4h crosses bullish or 1d reverses
            if inter_bull and rsi[i] < 35.0:
                desired_signal = 0.0
            if macro_bull:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            if desired_signal >= BASE_SIZE * 0.8:
                desired_signal = BASE_SIZE
            elif desired_signal >= REDUCED_SIZE * 0.8:
                desired_signal = REDUCED_SIZE
            else:
                desired_signal = REDUCED_SIZE * 0.5
        elif desired_signal < 0:
            if desired_signal <= -BASE_SIZE * 0.8:
                desired_signal = -BASE_SIZE
            elif desired_signal <= -REDUCED_SIZE * 0.8:
                desired_signal = -REDUCED_SIZE
            else:
                desired_signal = -REDUCED_SIZE * 0.5
        else:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Flip position
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals