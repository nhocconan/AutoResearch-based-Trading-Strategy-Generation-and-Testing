#!/usr/bin/env python3
"""
Experiment #021: 12h Donchian Breakout + Volume Confirmation + 1d SMA Trend

HYPOTHESIS: 20-period Donchian breakouts on 12h mark institutional moves.
Volume must confirm the breakout (ratio > 1.5x 20-period MA).
1d SMA(50) provides trend bias filter (only trade in direction of 1d trend).
1d SMA(200) for regime confirmation (avoid when price < SMA200 in bear).

WHY IT WORKS IN BULL AND BEAR:
- Bull: Long breakouts when price > 1d SMA50, volume confirms
- Bear: Short breakouts when price < 1d SMA50, or short rallies to SMA50 in bear regime
- 12h timeframe reduces noise and trade frequency vs 4h
- ATR-based stoploss of 2.5x protects against 2022-type crashes

TIMEFRAME: 12h primary
HTF: 1d for SMA50 trend + SMA200 regime
TARGET: 75-150 total trades over 4 years (19-37/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_vol_1d_sma_v1"
timeframe = "12h"
leverage = 1.0

def calculate_donchian(high, low, period=20):
    """Donchian Channel - returns upper and lower bands"""
    n = len(high)
    upper = np.full(n, np.nan, dtype=np.float64)
    middle = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
        middle[i] = (upper[i] + lower[i]) / 2.0
    
    return upper, middle, lower

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA50 for trend bias
    sma_1d_50_raw = pd.Series(df_1d['close'].values).rolling(window=50, min_periods=50).mean().values
    sma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, sma_1d_50_raw)
    
    # 1d SMA200 for regime (bear when price below)
    sma_1d_200_raw = pd.Series(df_1d['close'].values).rolling(window=200, min_periods=200).mean().values
    sma_1d_200_aligned = align_htf_to_ltf(prices, df_1d, sma_1d_200_raw)
    
    # Local 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    donch_upper, donch_middle, donch_lower = calculate_donchian(high, low, period=20)
    
    # Volume MA for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma_20 > 0, vol_ma_20, 1)
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking state
    position = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    entry_atr = 0.0
    
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 0 or np.isnan(donch_upper[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        if np.isnan(sma_1d_50_aligned[i]) or np.isnan(sma_1d_200_aligned[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # === TREND CONTEXT ===
        price_above_1d_sma50 = close[i] > sma_1d_50_aligned[i]
        price_above_1d_sma200 = close[i] > sma_1d_200_aligned[i]
        in_bear_regime = not price_above_1d_sma200
        
        # === VOLUME CONFIRMATION ===
        vol_confirm = vol_ratio[i] > 1.5
        
        # === DONCHIAN BREAKOUT ===
        # True breakout: close crosses above/below the 20-bar channel
        # Use close[i] vs donch_upper[i-1] to avoid look-ahead
        breakout_up = close[i] > donch_upper[i-1] and close[i-1] <= donch_upper[i-2]
        breakout_down = close[i] < donch_lower[i-1] and close[i-1] >= donch_lower[i-2]
        
        # === STOPLOSS CHECK ===
        if position != 0 and entry_atr > 0:
            if position > 0:
                # Long: stop if price falls 2.5 ATR below entry
                stop_loss = entry_price - 2.5 * entry_atr
                if low[i] <= stop_loss:
                    position = 0
                    entry_price = 0.0
                    entry_atr = 0.0
            else:
                # Short: stop if price rises 2.5 ATR above entry
                stop_loss = entry_price + 2.5 * entry_atr
                if high[i] >= stop_loss:
                    position = 0
                    entry_price = 0.0
                    entry_atr = 0.0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if position == 0:
            # === NEW LONG ===
            if breakout_up and vol_confirm and price_above_1d_sma50:
                desired_signal = SIZE
                position = 1
                entry_price = close[i]
                entry_atr = atr_14[i]
            
            # === NEW SHORT ===
            # In bear regime: short breakdowns
            # In any regime: short when price < SMA50 (counter-trend)
            elif breakout_down and vol_confirm:
                if in_bear_regime:
                    desired_signal = -SIZE
                    position = -1
                    entry_price = close[i]
                    entry_atr = atr_14[i]
                elif not price_above_1d_sma50:
                    # Counter-trend short in bull: be more selective
                    desired_signal = -SIZE
                    position = -1
                    entry_price = close[i]
                    entry_atr = atr_14[i]
        
        # === EXIT LOGIC ===
        if position != 0:
            if position > 0:
                # Long exit: price falls below 1d SMA50
                if not price_above_1d_sma50:
                    desired_signal = 0.0
                    position = 0
                    entry_price = 0.0
                    entry_atr = 0.0
            else:
                # Short exit: price rises above 1d SMA50
                if price_above_1d_sma50:
                    desired_signal = 0.0
                    position = 0
                    entry_price = 0.0
                    entry_atr = 0.0
        
        # Set signal
        if position > 0:
            desired_signal = SIZE
        elif position < 0:
            desired_signal = -SIZE
        else:
            desired_signal = 0.0
        
        signals[i] = desired_signal
    
    return signals