#!/usr/bin/env python3
"""
Experiment #008: 12h Donchian Breakout + RSI Extremes + Volume + 1w EMA200 Trend

HYPOTHESIS: Combining weekly trend direction (EMA200 on 1w) with 12h price action
captures multi-week swings. Long when price > 1w EMA200 + breaks 20-bar high with 
RSI oversold + volume spike. Short when price < 1w EMA200 + breaks 20-bar low with
RSI overbought + volume spike.

WHY IT WORKS IN BOTH MARKETS:
- Bull: Buy breakouts above weekly EMA (catches rallies)
- Bear: Short breakouts below weekly EMA (catches breakdowns)
- RSI extremes filter out weak signals
- ATR trailing stop limits 2022-style crash exposure

1w EMA200 provides the smoothest long-term trend signal (vs 1d which is noisier).
12h timeframe = ~730 bars/year = ~75-150 expected trades over 4 years.

Target: 75-150 total trades, 19-37/year. Max: 200.
Signal size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_rsi_vol_1w_v1"
timeframe = "12h"
leverage = 1.0

def calculate_rsi(prices, period=14):
    """Relative Strength Index"""
    n = len(prices)
    if n < period + 1:
        return np.full(n, np.nan)
    
    deltas = np.diff(prices, prepend=prices[0])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.zeros(n, dtype=np.float64)
    avg_loss = np.zeros(n, dtype=np.float64)
    
    avg_gain[period] = np.mean(gains[1:period+1])
    avg_loss[period] = np.mean(losses[1:period+1])
    
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i]) / period
    
    rs = np.divide(avg_gain, np.where(avg_loss > 0, avg_loss, 1e-10),
                   out=np.zeros(n), where=avg_loss > 0)
    rsi = 100 - (100 / (1 + rs))
    rsi[:period] = np.nan
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_adx(high, low, close, period=14):
    """Average Directional Index"""
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, 50.0)  # Neutral ADX
    
    # True range and directional movement
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    # Smoothed values
    atr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # DX calculation
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    
    for i in range(period, n):
        if atr_smooth[i] > 0:
            plus_di[i] = 100 * plus_dm_smooth[i] / atr_smooth[i]
            minus_di[i] = 100 * minus_dm_smooth[i] / atr_smooth[i]
            di_sum = plus_di[i] + minus_di[i]
            if di_sum > 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    # ADX
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    adx[:period * 2] = 50.0  # Neutral until warmed up
    
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1w = get_htf_data(prices, '1w')
    
    # 1w EMA200 for long-term trend
    ema_1w = pd.Series(df_1w['close'].values).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_12 = calculate_rsi(close, period=12)
    adx_14 = calculate_adx(high, low, close, period=14)
    
    # Volume ratio (20-bar MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # 12h Donchian channels (shift by 1 to avoid look-ahead)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_atr = 0.0
    
    warmup = 200  # Need enough for EMA200 alignment
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 0 or np.isnan(ema_1w_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === Weekly trend direction ===
        price_above_1w_ema = close[i] > ema_1w_aligned[i]
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.8
        
        # ADX regime filter (trend vs chop)
        adx_val = adx_14[i]
        is_trending = adx_val > 20  # trending market
        
        # === Entry logic ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Above weekly EMA + break high + RSI oversold + volume ===
            if price_above_1w_ema and is_trending and vol_spike:
                if rsi_12[i] < 40 and high[i] > donchian_high[i] and not np.isnan(donchian_high[i]):
                    desired_signal = SIZE
            
            # === SHORT: Below weekly EMA + break low + RSI overbought + volume ===
            if not price_above_1w_ema and is_trending and vol_spike:
                if rsi_12[i] > 60 and low[i] < donchian_low[i] and not np.isnan(donchian_low[i]):
                    desired_signal = -SIZE
        
        # === Trailing stop (2.0 ATR) ===
        if in_position:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.0 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
            else:
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.0 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
        
        # === Minimum hold (2 bars = 1 day) ===
        bars_held = i - entry_bar
        if in_position and bars_held >= 2:
            # RSI exit (momentum exhaustion)
            if position_side > 0 and rsi_12[i] > 70:
                desired_signal = 0.0
            if position_side < 0 and rsi_12[i] < 30:
                desired_signal = 0.0
        
        # === Update position ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_bar = i
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                entry_atr = atr_14[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals