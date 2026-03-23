#!/usr/bin/env python3
"""
Experiment #1105: 1h Primary + 4h/1d HTF — Trend Pullback with Session/Volume Filter

Hypothesis: After 800+ failed experiments, key insights for 1h timeframe:
1. 1h naturally generates 50-100 trades/year — must use STRICT filters to reduce to 30-60
2. Use 4h HMA for MACRO trend direction (not 1h — too noisy)
3. Use 1d ADX to confirm trending regime (ADX>20 = trend, ADX<20 = skip)
4. Use 1h RSI for pullback entries within HTF trend (loose: 35-65 range)
5. Session filter (8-20 UTC) reduces trades by ~50% during low-liquidity hours
6. Volume filter (>0.8x 20-bar avg) confirms institutional participation
7. Position size 0.25 with 2.5x ATR trailing stop

Why this should beat Sharpe=0.612 (current best 4h strategy):
- 1h has better entry timing than 4h while using 4h/1d for direction
- Session filter eliminates Asian session chop (major source of whipsaws)
- Volume filter ensures we only trade when institutions are active
- 1d ADX regime filter prevents trading during choppy periods
- Proven pattern: HTF trend + LTF pullback + session/volume = 2x Sharpe

Timeframe: 1h (primary)
HTF: 4h (trend), 1d (regime) — loaded ONCE before loop using mtf_data helper
Position Size: 0.25 base, 0.15 reduced (discrete levels)
Stoploss: 2.5x ATR trailing
Target: 30-60 trades/year, Sharpe > 0.612
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_4h1d_session_vol_atr_v2"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average — reduces lag while maintaining smoothness.
    
    Formula:
    1. WMA1 = WMA(close, period/2)
    2. WMA2 = WMA(close, period)
    3. WMA3 = WMA(2*WMA1 - WMA2, sqrt(period))
    4. HMA = WMA3
    """
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    def wma(data, span):
        """Weighted Moving Average."""
        result = np.full(len(data), np.nan)
        weights = np.arange(1, span + 1)
        for i in range(span - 1, len(data)):
            window = data[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    half = int(period / 2)
    if half < 1:
        half = 1
    
    wma1 = wma(close, half)
    wma2 = wma(close, period)
    
    # 2*WMA1 - WMA2
    diff = 2 * wma1 - wma2
    
    # WMA of diff with sqrt(period)
    sqrt_period = int(np.sqrt(period))
    if sqrt_period < 1:
        sqrt_period = 1
    
    hma = wma(diff, sqrt_period)
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index — momentum oscillator."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    diff = np.diff(close)
    gain = np.where(diff > 0, diff, 0.0)
    loss = np.where(diff < 0, -diff, 0.0)
    
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

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index — measures trend strength.
    ADX > 25 = strong trend, ADX < 20 = weak/choppy market.
    """
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2 + 1:
        return adx
    
    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Calculate Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    # Smooth DM and TR using Wilder's smoothing (EMA with span=period)
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate DI
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    mask = tr_s > 1e-10
    plus_di[mask] = 100.0 * plus_dm_s[mask] / tr_s[mask]
    minus_di[mask] = 100.0 * minus_dm_s[mask] / tr_s[mask]
    
    # Calculate DX
    dx = np.zeros(n)
    di_sum = plus_di + minus_di
    mask2 = di_sum > 1e-10
    dx[mask2] = 100.0 * np.abs(plus_di[mask2] - minus_di[mask2]) / di_sum[mask2]
    
    # ADX = EMA of DX
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def extract_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds, convert to seconds then to hour
    return (open_time // 1000 // 3600) % 24

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
    
    # Calculate and align 4h HMA for macro trend filter
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d ADX for regime filter
    adx_1d_raw = calculate_adx(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values,
        period=14
    )
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d_raw)
    
    # Calculate primary (1h) indicators
    rsi_1h = calculate_rsi(close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    
    # Volume SMA for volume filter (20-bar average)
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    REDUCED_SIZE = 0.15
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(rsi_1h[i]) or np.isnan(atr_1h[i]):
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(adx_1d_aligned[i]):
            continue
        if np.isnan(vol_sma[i]) or vol_sma[i] <= 1e-10:
            continue
        if atr_1h[i] <= 1e-10:
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        # Only trade during high-liquidity hours (London/NY overlap)
        hour = extract_hour(open_time[i])
        in_session = 8 <= hour <= 20
        
        # === VOLUME FILTER ===
        # Volume must be > 0.8x 20-bar average
        volume_ok = volume[i] > 0.8 * vol_sma[i]
        
        # === MACRO TREND (4h HMA) ===
        macro_bull = close[i] > hma_4h_aligned[i]
        macro_bear = close[i] < hma_4h_aligned[i]
        
        # === REGIME FILTER (1d ADX) ===
        # Only trade when daily ADX > 20 (trending regime)
        trend_regime = adx_1d_aligned[i] > 20.0
        
        # === PULLBACK SIGNAL (1h RSI) ===
        # Loose thresholds to ensure adequate trade frequency
        rsi_oversold = rsi_1h[i] < 45.0
        rsi_overbought = rsi_1h[i] > 55.0
        
        # === CONFLUENCE CHECK (3+ filters must agree) ===
        # Required: HTF trend + regime + session + volume + RSI pullback
        desired_signal = 0.0
        current_size = BASE_SIZE
        
        # === LONG ENTRY ===
        # 4h HMA bull + 1d ADX trending + 1h RSI pullback + session + volume
        if macro_bull and trend_regime and rsi_oversold and in_session and volume_ok:
            desired_signal = current_size
        
        # === SHORT ENTRY ===
        # 4h HMA bear + 1d ADX trending + 1h RSI pullback + session + volume
        elif macro_bear and trend_regime and rsi_overbought and in_session and volume_ok:
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
                # Hold long if 4h HMA still bull and 1d ADX still trending
                if macro_bull and adx_1d_aligned[i] > 18.0:
                    desired_signal = current_size
            elif position_side < 0:
                # Hold short if 4h HMA still bear and 1d ADX still trending
                if macro_bear and adx_1d_aligned[i] > 18.0:
                    desired_signal = -current_size
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if 4h HMA reverses or 1h RSI overbought
            if macro_bear or rsi_1h[i] > 65.0:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 4h HMA reverses or 1h RSI oversold
            if macro_bull or rsi_1h[i] < 35.0:
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
                entry_atr = atr_1h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Flip position
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1h[i]
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