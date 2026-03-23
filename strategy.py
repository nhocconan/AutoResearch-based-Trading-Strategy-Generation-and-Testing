#!/usr/bin/env python3
"""
Experiment #1155: 1h Primary + 4h/1d HTF — Multi-TF Trend + RSI Pullback + Volume + Session

Hypothesis: After 842+ failed experiments, clear patterns emerge:
- CRSI + Choppiness regime switching FAILS consistently (negative Sharpe #1144-#1148)
- Complex multi-regime logic causes 0 trades (#1148, #1150)
- SIMPLE trend + pullback WORKS (#1153 Sharpe=0.299, #1154 Sharpe=0.150)

For 1h timeframe, CRITICAL: use 4h/1d for SIGNAL DIRECTION, 1h only for ENTRY TIMING.
This gives HTF trade frequency (30-80/year) with 1h execution precision.

Strategy components:
1. 1d HMA(21) for MACRO trend direction (slowest, most reliable filter)
2. 4h HMA(21) for INTERMEDIATE trend confirmation
3. 1h RSI(7) pullback entries within HTF trend (RSI<35 long, RSI>65 short)
4. 1h Volume spike >1.5x SMA20 (confirms real moves, filters noise)
5. Session filter 8-20 UTC (avoid low liquidity Asian session)
6. 1h ATR(14) 2.5x trailing stop (protects gains, tight enough for 1h)
7. Position size 0.25 discrete (conservative for lower TF)

Why this should work on 1h:
- 1d/4h HTF filters ensure we only trade WITH macro trend
- RSI pullback entries catch retracements within trend (better than breakout)
- Volume + session filters reduce false signals during low liquidity
- Target: 40-80 trades/year on 1h (optimal for fee drag at this TF)

Timeframe: 1h (primary)
HTF: 4h, 1d — loaded ONCE before loop using mtf_data helper
Position Size: 0.25 base (discrete: 0.0, ±0.25)
Stoploss: 2.5x ATR trailing
Target: 40-80 trades/year, Sharpe > 0.612
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_pullback_4h1d_volume_session_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average — reduces lag while maintaining smoothness.
    HMA = WMA(2*WMA(n/2) - WMA(n))
    """
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # Calculate WMA
    def wma(series, span):
        weights = np.arange(1, span + 1)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    # HMA calculation
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            # Need to calculate WMA of the diff
            if i >= sqrt_period - 1:
                diff_window = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_window.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_window) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1)
                    hma[i] = np.sum(np.array(diff_window) * weights) / np.sum(weights)
    
    return hma

def calculate_rsi(close, period=14):
    """
    Relative Strength Index — momentum oscillator.
    RSI = 100 - 100/(1 + RS), RS = avg_gain/avg_loss
    """
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    # Use EMA for smoothing (Wilder's method)
    gain_series = pd.Series(np.concatenate([[0], gain]))
    loss_series = pd.Series(np.concatenate([[0], loss]))
    
    avg_gain = gain_series.ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = loss_series.ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.zeros(n)
    mask = avg_loss > 1e-10
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    
    rsi[period:] = 100.0 - 100.0 / (1.0 + rs[period:])
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement."""
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
    """Simple moving average of volume."""
    n = len(volume)
    vol_sma = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        vol_sma[i] = np.mean(volume[i - period + 1:i + 1])
    
    return vol_sma

def get_hour_from_timestamp(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
    return (open_time // (1000 * 3600)) % 24

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
    
    # Calculate and align 1d HMA for macro trend filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 4h HMA for intermediate trend filter
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate primary (1h) indicators
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=7)  # Faster RSI for pullback detection
    vol_sma = calculate_volume_sma(volume, period=20)
    
    # Also calculate 1h HMA for local trend confirmation
    hma_1h = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(rsi[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_4h_aligned[i]):
            continue
        if np.isnan(hma_1h[i]) or np.isnan(vol_sma[i]) or vol_sma[i] <= 1e-10:
            continue
        if atr[i] <= 1e-10:
            continue
        
        # Extract hour for session filter
        hour = get_hour_from_timestamp(open_time[i])
        
        # === SESSION FILTER (8-20 UTC) ===
        # Only trade during high liquidity hours (London + NY overlap)
        in_session = 8 <= hour <= 20
        
        # === MACRO TREND (1d HMA) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === INTERMEDIATE TREND (4h HMA) ===
        intermediate_bull = close[i] > hma_4h_aligned[i]
        intermediate_bear = close[i] < hma_4h_aligned[i]
        
        # === LOCAL TREND (1h HMA) ===
        local_bull = close[i] > hma_1h[i]
        local_bear = close[i] < hma_1h[i]
        
        # === VOLUME CONFIRMATION ===
        # Volume must be > 1.5x 20-bar SMA to confirm move is real
        volume_spike = volume[i] > 1.5 * vol_sma[i]
        
        # === RSI PULLBACK SIGNALS ===
        # Long: RSI < 35 (oversold pullback in uptrend)
        # Short: RSI > 65 (overbought pullback in downtrend)
        rsi_oversold = rsi[i] < 35.0
        rsi_overbought = rsi[i] > 65.0
        
        # === ENTRY CONDITIONS (ALL must align) ===
        desired_signal = 0.0
        
        # === LONG ENTRY ===
        # Macro bull + intermediate bull + RSI oversold + volume spike + in session
        if (macro_bull and intermediate_bull and rsi_oversold and 
            volume_spike and in_session):
            desired_signal = BASE_SIZE
        
        # === SHORT ENTRY ===
        # Macro bear + intermediate bear + RSI overbought + volume spike + in session
        elif (macro_bear and intermediate_bear and rsi_overbought and 
              volume_spike and in_session):
            desired_signal = -BASE_SIZE
        
        # === MACRO TREND REVERSAL EXIT ===
        # Exit if macro trend flips against position
        if in_position and position_side > 0 and macro_bear:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and macro_bull:
            desired_signal = 0.0
        
        # === INTERMEDIATE TREND WEAKNESS EXIT ===
        # Exit if intermediate trend flips (early warning)
        if in_position and position_side > 0 and intermediate_bear:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and intermediate_bull:
            desired_signal = 0.0
        
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
                # Hold long if macro, intermediate still bull
                if macro_bull and intermediate_bull:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if macro, intermediate still bear
                if macro_bear and intermediate_bear:
                    desired_signal = -BASE_SIZE
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE
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