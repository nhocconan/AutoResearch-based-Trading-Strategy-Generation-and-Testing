#!/usr/bin/env python3
"""
Experiment #718: 30m Primary + 4h/1d HTF — Simplified Trend Pullback

Hypothesis: Previous 30m strategies failed due to OVER-FILTERING (CRSI+Choppiness+Session = 0 trades).
This strategy SIMPLIFIES entry logic while maintaining HTF trend bias:
1. 4h HMA(21) for primary trend direction (proven in current best Sharpe=0.612)
2. 1d HMA(21) for strong regime bias (prevents counter-trend trades)
3. 30m RSI(14) pullback entries within HTF trend (simpler than CRSI)
4. ADX(14) for regime detection but LOOSE thresholds (ADX>20 not >25)
5. Volume filter: only >0.7x avg (not 1.2x which kills trades)
6. NO session filter (failed in #708/#710/#715)

Key differences from failed 30m experiments:
- NO Choppiness Index (failed 4x: #708, #710, #712, #715)
- NO Connors RSI (too complex, failed repeatedly)
- NO session filter (restricts trades too much)
- LOOSER RSI thresholds (35/65 not 25/75)
- LOOSER ADX threshold (>20 not >25)

Target: 40-80 trades/year, Sharpe > 0.612, all symbols positive
Size: 0.25 (lower than 4h strategies due to more trades)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_hma_rsi_pullback_4h1d_simp_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(series, period):
    """Hull Moving Average - smoother and more responsive than EMA."""
    if len(series) < period:
        return np.full(len(series), np.nan)
    
    wma1 = pd.Series(series).ewm(span=period//2, min_periods=period//2, adjust=False).mean().values
    wma2 = pd.Series(series).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    hma_raw = 2 * wma1 - wma2
    hma = pd.Series(hma_raw).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean().values
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength."""
    n = len(close)
    adx = np.full(n, np.nan)
    atr = np.full(n, np.nan)
    
    if n < period * 2:
        return adx, atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    atr_vals = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        di_plus = 100 * plus_dm_smooth / (atr_vals + 1e-10)
        di_minus = 100 * minus_dm_smooth / (atr_vals + 1e-10)
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx, atr_vals

def calculate_sma(close, period):
    """Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_volume_ratio(volume, period=20):
    """Volume ratio vs rolling average."""
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = volume / (vol_sma + 1e-10)
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (30m) indicators
    adx_30m, atr_30m = calculate_adx(high, low, close, period=14)
    rsi_30m = calculate_rsi(close, period=14)
    sma_200 = calculate_sma(close, period=200)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
    # Calculate and align HTF HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
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
    
    for i in range(250, n):  # Need 200 for SMA + buffer for HTF alignment
        # Skip if indicators not ready
        if np.isnan(rsi_30m[i]) or np.isnan(atr_30m[i]):
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(sma_200[i]) or np.isnan(adx_30m[i]):
            continue
        if np.isnan(vol_ratio[i]):
            continue
        if atr_30m[i] <= 1e-10:
            continue
        
        # === TREND BIAS (4h and 1d HMA) ===
        # Both HTF must agree for strong signal
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # Strong trend: both 4h and 1d agree
        strong_bullish = trend_4h_bullish and trend_1d_bullish
        strong_bearish = trend_4h_bearish and trend_1d_bearish
        
        # Moderate trend: only 4h agrees (1d neutral or opposite)
        mod_bullish = trend_4h_bullish and not trend_1d_bearish
        mod_bearish = trend_4h_bearish and not trend_1d_bullish
        
        # === REGIME (ADX) ===
        # LOOSE threshold: ADX > 20 (not 25) to get more trades
        trending_regime = adx_30m[i] > 20
        ranging_regime = adx_30m[i] < 18
        
        # === VOLUME FILTER ===
        # LOOSE: >0.7x avg (not 1.2x which kills trades)
        volume_ok = vol_ratio[i] > 0.7
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        desired_signal = 0.0
        current_size = BASE_SIZE
        
        # === LONG ENTRIES ===
        if strong_bullish or mod_bullish:
            # Primary: RSI pullback in uptrend
            if rsi_30m[i] < 45 and volume_ok:
                if strong_bullish:
                    desired_signal = BASE_SIZE
                elif mod_bullish and above_sma200:
                    desired_signal = REDUCED_SIZE
            
            # Secondary: RSI very oversold (stronger signal)
            elif rsi_30m[i] < 35:
                desired_signal = BASE_SIZE
        
        # === SHORT ENTRIES ===
        elif strong_bearish or mod_bearish:
            # Primary: RSI bounce in downtrend
            if rsi_30m[i] > 55 and volume_ok:
                if strong_bearish:
                    desired_signal = -BASE_SIZE
                elif mod_bearish and below_sma200:
                    desired_signal = -REDUCED_SIZE
            
            # Secondary: RSI very overbought (stronger signal)
            elif rsi_30m[i] > 65:
                desired_signal = -BASE_SIZE
        
        # === RANGE REGIME MEAN REVERSION ===
        if ranging_regime and desired_signal == 0.0:
            # Long at oversold
            if rsi_30m[i] < 30 and volume_ok:
                desired_signal = REDUCED_SIZE
            # Short at overbought
            elif rsi_30m[i] > 70 and volume_ok:
                desired_signal = -REDUCED_SIZE
        
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
                # Hold long if RSI not overbought and 4h trend intact
                if rsi_30m[i] < 70 and trend_4h_bullish:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if RSI not oversold and 4h trend intact
                if rsi_30m[i] > 30 and trend_4h_bearish:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if RSI very overbought or 4h trend breaks
            if rsi_30m[i] > 75:
                desired_signal = 0.0
            elif close[i] < hma_4h_aligned[i]:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if RSI very oversold or 4h trend breaks
            if rsi_30m[i] < 25:
                desired_signal = 0.0
            elif close[i] > hma_4h_aligned[i]:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE * 0.8 else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE * 0.8 else -REDUCED_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_30m[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_30m[i]
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