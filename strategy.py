#!/usr/bin/env python3
"""
Experiment #021: 4h Donchian Breakout + HMA + Volume (Proven Pattern)

HYPOTHESIS: The HMA+Donchian+Volume combination is the MOST validated pattern
in our DB (test Sharpe 1.38-1.46 on SOL). 4h timeframe is optimal - slow enough
to avoid fee drag, fast enough to catch meaningful breakouts. 12h HTF provides
trend confirmation. This is NOT a new strategy - it's implementing the proven
winning formula with disciplined entry to avoid overtrading.

KEY INSIGHT: DB shows mtf_4h_hma_donchian_volume_rsi_12h_atr_v1 (95 trades, 
Sharpe 1.38) works. We're copying that structure with tighter entry filters
to target 75-150 trades.

TIMEFRAME: 4h primary
HTF: 12h for trend, 1d for regime
TARGET: 75-150 total trades over 4 years (19-37/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_donchian_vol_rsi_12h_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - returns upper and lower bands"""
    n = len(high)
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    mid = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
        mid[i] = (upper[i] + lower[i]) / 2.0
    
    return upper, lower, mid

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # 12h HMA for trend
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # 1d HMA for regime check
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Local indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    donch_upper, donch_lower, donch_mid = calculate_donchian(high, low, period=20)
    
    # Volume MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # RSI
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, min_periods=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = (100 - (100 / (1 + rs))).values
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND (12h HMA) ===
        price_above_12h_hma = close[i] > hma_12h_aligned[i]
        bullish_trend = price_above_12h_hma
        bearish_trend = not price_above_12h_hma
        
        # === REGIME (1d HMA) - filter extreme bear ===
        price_above_1d_hma = not np.isnan(hma_1d_aligned[i]) and close[i] > hma_1d_aligned[i]
        price_below_1d_hma = not np.isnan(hma_1d_aligned[i]) and close[i] < hma_1d_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5  # Stronger than typical 1.3
        
        # === RSI ===
        rsi_val = rsi[i]
        
        # === DONCHIAN BREAKOUT ===
        # Breakout: close crosses above/below the channel
        breakout_up = (close[i] > donch_upper[i]) and (close[i-1] <= donch_upper[i-1] if i > 1 else True)
        breakout_down = (close[i] < donch_lower[i]) and (close[i-1] >= donch_lower[i-1] if i > 1 else True)
        
        # Already broken: price clearly outside channel
        price_above_upper = close[i] > donch_upper[i]
        price_below_lower = close[i] < donch_lower[i]
        
        desired_signal = 0.0
        
        if not in_position:
            # === LONG ENTRY ===
            # Need: breakout OR price above channel + volume + RSI not overbought + bullish 12h trend
            if breakout_up or price_above_upper:
                if vol_spike and rsi_val < 70 and bullish_trend:
                    # In bear regime (price below 1d HMA), only enter on strong reversal
                    if price_above_1d_hma or rsi_val < 40:
                        desired_signal = SIZE
            
            # === SHORT ENTRY ===
            # Need: breakout OR price below channel + volume + RSI not oversold + bearish 12h trend
            if breakout_down or price_below_lower:
                if vol_spike and rsi_val > 30 and bearish_trend:
                    # In bull regime (price above 1d HMA), only enter on strong reversal
                    if price_below_1d_hma or rsi_val > 60:
                        desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            if low[i] < trailing_stop:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            if high[i] > trailing_stop:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === EXIT ===
        exit_triggered = False
        
        if in_position and position_side > 0:
            # Long exit: price breaks below lower channel OR RSI oversold
            if price_below_lower:
                exit_triggered = True
            if rsi_val < 30:
                exit_triggered = True
        
        if in_position and position_side < 0:
            # Short exit: price breaks above upper channel OR RSI overbought
            if price_above_upper:
                exit_triggered = True
            if rsi_val > 70:
                exit_triggered = True
        
        if exit_triggered:
            desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
            # Same direction: hold
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals