#!/usr/bin/env python3
"""
Experiment #012 - v2: 12h Williams %R + ATR Volatility Expansion + 1d Trend

HYPOTHESIS: Williams %R extremes (< -80 long, > -20 short) catch reversal points.
ATR expansion filter (ATR(14)/ATR(30) > 1.15) ensures we only trade during volatile
moves, avoiding the 2022 choppy bear market. 1d SMA(50) as trend filter keeps us
aligned with higher timeframe direction. 12h is slow enough to avoid overtrading
while capturing meaningful reversals. Williams %R is proven on crypto (reversals
are sharper than stocks).

TIMEFRAME: 12h primary
HTF: 1d for trend SMA, 1w for regime check
TARGET: 75-200 total trades over 4 years (19-50/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_williamsr_atr_exp_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_williams_r(high, low, close, period=14):
    """Williams %R - momentum oscillator"""
    n = len(close)
    wr = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        if highest_high != lowest_low:
            wr[i] = -100 * (highest_high - close[i]) / (highest_high - lowest_low)
        else:
            wr[i] = -50  # neutral when no range
    
    return wr

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

def calculate_sma(values, period):
    """Simple Moving Average"""
    return pd.Series(values).rolling(window=period, min_periods=period).mean().values

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength"""
    n = len(close)
    if n < period * 2:
        return np.full(n, np.nan)
    
    # True Range components
    tr_list = []
    plus_dm_list = []
    minus_dm_list = []
    
    for i in range(1, n):
        tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        plus_dm = max(high[i] - high[i-1], 0) if (high[i] - high[i-1]) > (low[i-1] - low[i]) else 0
        minus_dm = max(low[i-1] - low[i], 0) if (low[i-1] - low[i]) > (high[i] - high[i-1]) else 0
        tr_list.append(tr)
        plus_dm_list.append(plus_dm)
        minus_dm_list.append(minus_dm)
    
    tr_arr = np.array(tr_list)
    plus_dm_arr = np.array(plus_dm_list)
    minus_dm_arr = np.array(minus_dm_list)
    
    # Smooth with EMA
    atr_smooth = pd.Series(tr_arr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm_arr).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm_arr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # DX calculation
    di_plus = np.zeros(len(atr_smooth))
    di_minus = np.zeros(len(atr_smooth))
    dx = np.zeros(len(atr_smooth))
    
    for i in range(len(atr_smooth)):
        if atr_smooth[i] > 0:
            di_plus[i] = 100 * plus_dm_smooth[i] / atr_smooth[i]
            di_minus[i] = 100 * minus_dm_smooth[i] / atr_smooth[i]
            di_sum = di_plus[i] + di_minus[i]
            if di_sum > 0:
                dx[i] = 100 * abs(di_plus[i] - di_minus[i]) / di_sum
    
    # ADX as smoothed DX
    adx = np.full(n, np.nan)
    adx_values = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    for i in range(period * 2, n):
        adx[i] = adx_values[i - 1] if i > 0 else np.nan
    
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # 1d SMA(50) for trend filter
    sma_1d_raw = calculate_sma(df_1d['close'].values, period=50)
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d_raw)
    
    # 1w SMA(21) for regime
    sma_1w_raw = calculate_sma(df_1w['close'].values, period=21)
    sma_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_1w_raw)
    
    # Calculate local 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_30 = calculate_atr(high, low, close, period=30)
    
    # ATR expansion ratio
    atr_expansion = atr_14 / np.where(atr_30 > 0, atr_30, 1)
    
    # Williams %R
    williams_r = calculate_williams_r(high, low, close, period=14)
    
    # ADX for trend strength
    adx_14 = calculate_adx(high, low, close, period=14)
    
    # Volume MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = 60
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(williams_r[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(sma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # Current indicator values
        wr = williams_r[i]
        atr_exp = atr_expansion[i]
        adx = adx_14[i]
        
        # === TREND FILTER (1d SMA) ===
        price_above_1d_sma = close[i] > sma_1d_aligned[i]
        price_below_1d_sma = close[i] < sma_1d_aligned[i]
        
        # === REGIME (1w SMA) ===
        price_above_1w_sma = close[i] > sma_1w_aligned[i] if not np.isnan(sma_1w_aligned[i]) else True
        price_below_1w_sma = close[i] < sma_1w_aligned[i] if not np.isnan(sma_1w_aligned[i]) else False
        
        # === VOLATILITY FILTER ===
        # Only enter during ATR expansion (volatility breakout)
        vol_expansion = atr_exp > 1.15
        
        # === TREND STRENGTH FILTER ===
        # ADX > 20 = trending (don't fight trend in ranging)
        trending = adx > 20 if not np.isnan(adx) else True
        
        # === VOLUME CONFIRMATION ===
        vol_confirm = vol_ratio[i] > 1.0
        
        # === ENTRY CONDITIONS ===
        # Long: Williams %R < -80 (extreme oversold) + price above 1d SMA + vol expansion
        # Short: Williams %R > -20 (extreme overbought) + price below 1d SMA + vol expansion
        
        desired_signal = 0.0
        
        if not in_position:
            # === NEW LONG ===
            # Oversold + trend aligned + volatility expansion
            if wr < -80:
                if price_above_1d_sma and vol_expansion and (trending or price_above_1w_sma):
                    desired_signal = SIZE
            
            # === NEW SHORT ===
            # Overbought + trend aligned + volatility expansion
            if wr > -20:
                if price_below_1d_sma and vol_expansion and (trending or price_below_1w_sma):
                    desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === EXIT: Opposite Williams %R extreme ===
        exit_triggered = False
        
        if in_position and position_side > 0:
            # Long exit: Williams %R reaches overbought OR price crosses below 1d SMA
            if wr > -20:
                exit_triggered = True
            if price_below_1d_sma:
                exit_triggered = True
        
        if in_position and position_side < 0:
            # Short exit: Williams %R reaches oversold OR price crosses above 1d SMA
            if wr < -80:
                exit_triggered = True
            if price_above_1d_sma:
                exit_triggered = True
        
        if exit_triggered:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                entry_bar = i
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
            else:
                # Same direction - maintain position
                pass
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                entry_bar = 0
        
        signals[i] = desired_signal
    
    return signals