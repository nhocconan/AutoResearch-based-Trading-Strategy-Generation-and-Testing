#!/usr/bin/env python3
"""
Experiment #378: 30m Primary + 4h/1d HTF — Relaxed Confluence with Regime Adaptive Entries

Hypothesis: 30m timeframe can work IF we use HTF for direction and 30m only for timing.
Key insight from failures: too strict filters = 0 trades. Need RELAXED thresholds.

STRATEGY DESIGN:
1. 4h HMA(21) = trend bias (direction only, not entry trigger)
2. 1d ADX(14) = regime filter (ADX>25 trend, ADX<20 range)
3. 30m RSI(14) = entry timing (relaxed: <40 long, >60 short — NOT 20/80)
4. 30m ATR(14) = stoploss (2.5*ATR trailing)
5. Volume filter: lenient (0.6x 20-bar avg, not 1.0x)
6. Session: 6-22 UTC (wider window for crypto 24/7)

POSITION SIZING: 0.25 (25% capital) — smaller for 30m to reduce fee drag
TARGET: 40-80 trades/year on 30m (NOT >100 or fees kill profit)

WHY THIS MIGHT WORK:
- Relaxed RSI thresholds ensure we get trades (failed exp #368 had 0 trades)
- HTF bias prevents counter-trend entries in strong trends
- Regime adaptive: different logic for trending vs ranging markets
- ATR stoploss protects from 2022-style crashes

CRITICAL: Call get_htf_data() ONCE before loop, use aligned arrays inside.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_regime_rsi_hma_4h1d_relaxed_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma_half = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    hull = 2.0 * wma_half - wma_full
    hma = hull.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

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
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0).values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    return adx.fillna(20.0).values

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
    
    # Calculate 30m indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    sma_200 = calculate_sma(close, period=200)
    
    # Calculate 30m volume MA for filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate and align HTF HMA for bias (4h)
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align HTF ADX for regime (1d)
    adx_1d_raw = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # 25% position size for 30m (target 40-80 trades/year)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(rsi_14[i]) or np.isnan(sma_200[i]):
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(adx_1d_aligned[i]):
            continue
        if np.isnan(vol_ma_20[i]) or vol_ma_20[i] <= 1e-10:
            continue
        
        # Extract hour from open_time for session filter
        # open_time is in milliseconds, convert to hour
        hour_utc = (open_time[i] // (1000 * 60 * 60)) % 24
        
        # === SESSION FILTER (6-22 UTC) ===
        in_session = 6 <= hour_utc <= 22
        
        # === VOLUME FILTER (lenient: 0.6x avg) ===
        volume_ok = volume[i] >= 0.6 * vol_ma_20[i]
        
        # === HTF BIAS (4h HMA) ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === REGIME DETECTION (1d ADX) ===
        is_trending = adx_1d_aligned[i] > 25.0  # ADX > 25 = trending
        is_ranging = adx_1d_aligned[i] < 20.0   # ADX < 20 = ranging
        
        # === SMA200 FILTER ===
        price_above_sma200 = close[i] > sma_200[i]
        price_below_sma200 = close[i] < sma_200[i]
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        if is_trending:
            # TREND REGIME: Follow HTF bias, enter on RSI pullback
            # Relaxed RSI: <45 for long pullback, >55 for short pullback
            
            rsi_pullback_long = rsi_14[i] < 45.0
            rsi_pullback_short = rsi_14[i] > 55.0
            
            # Long: 4h bullish + SMA200 bullish + RSI pullback + session + volume
            if price_above_hma_4h and price_above_sma200 and rsi_pullback_long and in_session and volume_ok:
                desired_signal = BASE_SIZE
            
            # Short: 4h bearish + SMA200 bearish + RSI pullback + session + volume
            elif price_below_hma_4h and price_below_sma200 and rsi_pullback_short and in_session and volume_ok:
                desired_signal = -BASE_SIZE
        
        elif is_ranging:
            # RANGE REGIME: Mean reversion at extremes
            # Relaxed RSI: <40 oversold, >60 overbought
            
            rsi_oversold = rsi_14[i] < 40.0
            rsi_overbought = rsi_14[i] > 60.0
            
            # Long: 4h bullish + RSI oversold + session + volume
            if price_above_hma_4h and rsi_oversold and in_session and volume_ok:
                desired_signal = BASE_SIZE
            
            # Short: 4h bearish + RSI overbought + session + volume
            elif price_below_hma_4h and rsi_overbought and in_session and volume_ok:
                desired_signal = -BASE_SIZE
        
        else:
            # NEUTRAL REGIME: Only strong RSI extremes
            rsi_oversold = rsi_14[i] < 35.0
            rsi_overbought = rsi_14[i] > 65.0
            
            if price_above_hma_4h and rsi_oversold and in_session and volume_ok:
                desired_signal = BASE_SIZE
            elif price_below_hma_4h and rsi_overbought and in_session and volume_ok:
                desired_signal = -BASE_SIZE
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
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
        
        # === RSI EXIT (mean reversion complete) ===
        if in_position and position_side > 0 and rsi_14[i] > 65:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and rsi_14[i] < 35:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position unless clear exit trigger ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and price_above_hma_4h:
                desired_signal = BASE_SIZE
            elif position_side < 0 and price_below_hma_4h:
                desired_signal = -BASE_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
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