#!/usr/bin/env python3
"""
Experiment #118: 30m Primary + 4h/1d HTF — HMA Trend + ADX Regime + RSI Pullback

Hypothesis: After 117 experiments, the pattern is clear for lower timeframes:
- 30m needs VERY strict filters to avoid fee drag (>100 trades/yr = death)
- BUT filters must still allow trades (recent exps #106-117 have Sharpe=0.000 = 0 trades!)
- Key insight: Use HTF (1d/4h) for DIRECTION, 30m only for ENTRY TIMING
- This gives HTF trade frequency with 30m execution precision

Strategy design:
1. 1d HMA(21) = major trend bias (price above/below)
2. 4h ADX(14) = trend strength filter (ADX>25 = trend, ADX<20 = range)
3. 30m RSI(14) = entry trigger with MODERATE thresholds (35/65, not extreme)
4. Session filter: 8-20 UTC only (reduces overnight noise)
5. Volume filter: volume > 0.7x 20-bar avg (confirms participation)
6. Position size: 0.25 (25% - conservative for 30m)
7. Stoploss: 2.5x ATR trailing (tighter for lower TF)

Why this should work:
- 1d HMA bias ensures we trade WITH major trend (proven in exp #100)
- 4h ADX regime prevents mean-reversion entries during strong trends
- RSI 35/65 thresholds are loose enough to generate trades (unlike 25/75 which failed)
- Session+volume filters reduce trade count to 30-80/yr target
- Smaller size (0.25 vs 0.35) controls drawdown on lower TF

Target: Sharpe>0.351, DD>-40%, trades>=30 train, trades>=3 test, trades<100/yr
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_hma_adx_rsi_session_4h1d_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average - more responsive than EMA, less lag
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            result[i] = np.sum(series[i-span+1:i+1] * weights) / np.sum(weights)
        return result
    
    close_series = pd.Series(close)
    wma_half = wma(close, period // 2)
    wma_full = wma(close, period)
    
    hma_raw = 2.0 * wma_half - wma_full
    hma = wma(hma_raw, int(np.sqrt(period)))
    
    return hma

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index - measures trend strength
    ADX > 25 = trending, ADX < 20 = ranging
    """
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, np.nan)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    # Smooth with EMA
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    for i in range(period, n):
        if tr_s[i] > 1e-10:
            plus_di[i] = 100.0 * plus_dm_s[i] / tr_s[i]
            minus_di[i] = 100.0 * minus_dm_s[i] / tr_s[i]
    
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Extract hour from open_time for session filter
    # open_time is in milliseconds, convert to hour
    hours = np.zeros(n, dtype=int)
    for i in range(n):
        ts_ms = prices["open_time"].iloc[i]
        hours[i] = (ts_ms // (1000 * 60 * 60)) % 24
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align 1d HMA for major trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 4h ADX for regime detection
    adx_4h_raw = calculate_adx(df_4h['high'].values, df_4h['low'].values, df_4h['close'].values, period=14)
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h_raw)
    
    # Calculate primary (30m) indicators
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    # Volume moving average for volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size (conservative for 30m)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]) or np.isnan(hma_1d_aligned[i]) or np.isnan(adx_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(vol_ma[i]) or vol_ma[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d HMA) ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === REGIME FILTER (4h ADX) ===
        # ADX > 25 = trending (follow trend)
        # ADX < 20 = ranging (can mean revert)
        # ADX 20-25 = neutral (allow both)
        adx_val = adx_4h_aligned[i]
        is_trending = adx_val > 25.0
        is_ranging = adx_val < 20.0
        
        # === RSI ENTRY (MODERATE thresholds for trade generation) ===
        # Long: RSI < 50 (pullback in uptrend) or RSI < 35 (oversold in range)
        # Short: RSI > 50 (pullback in downtrend) or RSI > 65 (overbought in range)
        rsi_val = rsi[i]
        
        rsi_ok_long = False
        rsi_ok_short = False
        
        if is_trending:
            # In trend: allow pullback entries (RSI 35-65 range)
            rsi_ok_long = rsi_val < 55.0
            rsi_ok_short = rsi_val > 45.0
        else:
            # In range: require more extreme RSI
            rsi_ok_long = rsi_val < 45.0
            rsi_ok_short = rsi_val > 55.0
        
        # === SESSION FILTER (8-20 UTC only) ===
        in_session = (hours[i] >= 8) and (hours[i] <= 20)
        
        # === VOLUME FILTER ===
        vol_ok = volume[i] > 0.7 * vol_ma[i]
        
        # === DESIRED SIGNAL ===
        # LONG: 1d bull + (trending+RSI<55 OR ranging+RSI<45) + session + volume
        # SHORT: 1d bear + (trending+RSI>45 OR ranging+RSI>55) + session + volume
        desired_signal = 0.0
        
        if htf_bull and rsi_ok_long and in_session and vol_ok:
            desired_signal = SIZE
        elif htf_bear and rsi_ok_short and in_session and vol_ok:
            desired_signal = -SIZE
        
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
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
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
        
        signals[i] = final_signal
    
    return signals