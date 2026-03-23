#!/usr/bin/env python3
"""
Experiment #1175: 1h Primary + 4h/1d HTF — HMA Trend + RSI Pullback + Volume + Session

Hypothesis: After 859+ failed experiments, clear patterns emerge for 1h timeframe:
- 1h strategies fail with 0 trades when filters are too strict (#1165, #1168, #1170)
- Key insight: Use 4h/1d for TREND DIRECTION, 1h only for ENTRY TIMING
- RSI pullback (not extreme) works better than CRSI for 1h frequency
- Volume threshold 0.8x (not 1.5x) ensures enough trades while filtering noise
- Session filter 8-20 UTC captures high-liquidity periods (London/NY overlap)
- Position size 0.25 (smaller than 4h) to minimize fee drag on more frequent trades
- Target: 40-60 trades/year on 1h (not 200+ which kills profit via fees)

Why this should beat Sharpe=0.612:
- Simpler entry logic = more trades (avoid 0-trade failures)
- 4h HMA provides macro trend filter without being too slow
- RSI 35-45 pullback in uptrend catches dips, not extremes
- Volume 0.8x confirms participation without being too restrictive
- Session filter reduces whipsaw during low-liquidity hours
- ATR 2.5x trailing stop appropriate for 1h volatility

Timeframe: 1h (primary)
HTF: 4h (trend), 1d (macro filter) — loaded ONCE before loop using mtf_data helper
Position Size: 0.25 base (discrete: 0.0, ±0.25)
Stoploss: 2.5x ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_pullback_4h1d_session_vol_atr_v1"
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
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
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
    RSI 35-45 = bullish pullback, RSI 55-65 = bearish pullback
    """
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = loss_smooth > 1e-10
    rs = np.zeros(n)
    rs[mask] = gain_smooth[mask] / loss_smooth[mask]
    
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    
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

def calculate_volume_ratio(volume, period=20):
    """
    Volume ratio — current volume vs average.
    Returns ratio (1.0 = average, >1.0 = above average)
    """
    n = len(volume)
    vol_ratio = np.full(n, np.nan)
    
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    
    for i in range(period - 1, n):
        if vol_avg[i] > 1e-10:
            vol_ratio[i] = volume[i] / vol_avg[i]
    
    return vol_ratio

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
    
    # Calculate and align 4h HMA for trend filter
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for macro filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (1h) indicators
    atr = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    hma_1h = calculate_hma(close, period=21)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
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
        if np.isnan(atr[i]) or np.isnan(rsi_1h[i]) or np.isnan(hma_1h[i]):
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(vol_ratio[i]) or atr[i] <= 1e-10:
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        # Extract hour from open_time (milliseconds timestamp)
        hour_utc = (open_time[i] // 3600000) % 24
        in_session = 8 <= hour_utc <= 20
        
        # === MACRO TREND (1d HMA) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === INTERMEDIATE TREND (4h HMA) ===
        trend_bull = close[i] > hma_4h_aligned[i]
        trend_bear = close[i] < hma_4h_aligned[i]
        
        # === LOCAL TREND (1h HMA) ===
        local_bull = close[i] > hma_1h[i]
        local_bear = close[i] < hma_1h[i]
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = vol_ratio[i] >= 0.8  # 80% of average is enough
        
        # === RSI PULLBACK (not extreme) ===
        # Long: RSI 35-50 in uptrend (pullback, not oversold)
        # Short: RSI 50-65 in downtrend (pullback, not overbought)
        rsi_pullback_long = 35.0 <= rsi_1h[i] <= 50.0
        rsi_pullback_short = 50.0 <= rsi_1h[i] <= 65.0
        
        # === ENTRY CONDITIONS ===
        desired_signal = 0.0
        
        # === LONG ENTRY ===
        # Macro bull + 4h trend bull + RSI pullback + volume + session
        # Relaxed: only need 4h trend (not all 3 timeframes aligned)
        if macro_bull and trend_bull and rsi_pullback_long and vol_confirmed and in_session:
            desired_signal = BASE_SIZE
        
        # === SHORT ENTRY ===
        # Macro bear + 4h trend bear + RSI pullback + volume + session
        if macro_bear and trend_bear and rsi_pullback_short and vol_confirmed and in_session:
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
                # Hold long if macro and 4h trend still bull
                if macro_bull and trend_bull:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if macro and 4h trend still bear
                if macro_bear and trend_bear:
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