#!/usr/bin/env python3
"""
Experiment #1158: 30m Primary + 4h/1d HTF — Triple HMA Trend + RSI Pullback + Session Filter

Hypothesis: After 845+ failed experiments, clear patterns emerge:
- CRSI + Choppiness regime FAILS consistently (Sharpe=0.000 or negative in #1146-#1150)
- Simple HMA trend + volume breakout WORKS (#1154 Sharpe=0.150)
- Lower TF (30m) MUST have strict filters to avoid fee drag (>100 trades/year = death)

This strategy uses PROVEN components with NEW combination for 30m:
1. 1d HMA(21) for MACRO trend direction (slowest, most reliable — avoid 2022 whipsaw)
2. 4h HMA(21) for INTERMEDIATE trend confirmation (aligns with macro)
3. 30m RSI(14) pullback entry (buy dips in uptrend, sell rallies in downtrend)
4. 30m Volume spike (>1.3x SMA20) — confirms real moves, filters noise
5. Session filter (8-20 UTC) — avoids low-liquidity Asian overnight hours
6. 30m ATR(14) 2.5x trailing stop — wider stop for lower TF noise
7. Position size 0.20 (smaller for lower TF to reduce fee impact)

Why this should beat Sharpe=0.612:
- 1d HMA prevents trading against macro trend (major failure mode in 2022)
- 4h HMA adds intermediate confirmation (reduces false signals)
- RSI pullback enters at better prices than breakout (proven in #1153)
- Session filter cuts 40% of low-quality overnight trades
- Target: 40-70 trades/year on 30m (optimal for fee drag at this TF)

Timeframe: 30m (primary)
HTF: 4h and 1d — loaded ONCE before loop using mtf_data helper
Position Size: 0.20 base (discrete: 0.0, ±0.20)
Stoploss: 2.5x ATR trailing (wider for 30m noise)
Target: 40-70 trades/year, Sharpe > 0.612
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_triple_hma_rsi_pullback_session_4h1d_atr_v1"
timeframe = "30m"
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
    RSI = 100 - (100 / (1 + RS))
    RS = avg_gain / avg_loss
    """
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Pad to match length
    gain = np.concatenate([[0], gain])
    loss = np.concatenate([[0], loss])
    
    # EMA smoothing
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate RSI
    mask = loss_smooth > 1e-10
    rs = np.zeros(n)
    rs[mask] = gain_smooth[mask] / loss_smooth[mask]
    rsi[mask] = 100.0 - (100.0 / (1.0 + rs[mask]))
    
    # Handle division by zero (all gains)
    rsi[loss_smooth <= 1e-10] = 100.0
    
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

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
    return pd.to_datetime(open_time, unit='ms').hour

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
    
    # Calculate primary (30m) indicators
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    vol_sma = calculate_volume_sma(volume, period=20)
    
    # Also calculate 30m HMA for local trend
    hma_30m = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.20
    
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
        if np.isnan(hma_30m[i]) or np.isnan(vol_sma[i]):
            continue
        if vol_sma[i] <= 1e-10 or atr[i] <= 1e-10:
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        # Only trade during high-liquidity hours (avoid Asian overnight)
        utc_hour = get_utc_hour(open_time[i])
        in_session = 8 <= utc_hour <= 20
        
        # === MACRO TREND (1d HMA) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === INTERMEDIATE TREND (4h HMA) ===
        inter_bull = close[i] > hma_4h_aligned[i]
        inter_bear = close[i] < hma_4h_aligned[i]
        
        # === LOCAL TREND (30m HMA) ===
        local_bull = close[i] > hma_30m[i]
        local_bear = close[i] < hma_30m[i]
        
        # === VOLUME CONFIRMATION ===
        # Volume must be > 1.3x 20-bar SMA to confirm move is real
        volume_spike = volume[i] > 1.3 * vol_sma[i]
        
        # === RSI PULLBACK ENTRY ===
        # Long: RSI < 40 (oversold pullback) in uptrend
        # Short: RSI > 60 (overbought rally) in downtrend
        rsi_oversold = rsi[i] < 40.0
        rsi_overbought = rsi[i] > 60.0
        
        # === ENTRY CONDITIONS (3+ CONFLUENCE REQUIRED) ===
        desired_signal = 0.0
        
        # === LONG ENTRY ===
        # Macro bull + inter bull + RSI oversold + volume spike + in session
        if macro_bull and inter_bull and rsi_oversold and volume_spike and in_session:
            desired_signal = BASE_SIZE
        
        # === SHORT ENTRY ===
        # Macro bear + inter bear + RSI overbought + volume spike + in session
        elif macro_bear and inter_bear and rsi_overbought and volume_spike and in_session:
            desired_signal = -BASE_SIZE
        
        # === MACRO TREND REVERSAL EXIT ===
        if in_position and position_side > 0 and macro_bear:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and macro_bull:
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
                # Hold long if macro and inter still bull
                if macro_bull and inter_bull:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if macro and inter still bear
                if macro_bear and inter_bear:
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