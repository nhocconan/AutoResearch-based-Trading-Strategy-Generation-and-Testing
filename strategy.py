#!/usr/bin/env python3
"""
Experiment #136: 12h Donchian Breakout + 1d Volume Spike + ATR Regime Filter

HYPOTHESIS: Donchian channel breakouts on 12h timeframe, confirmed by 1d volume spikes (>2.0x average) 
and filtered by ATR-based volatility regime (ATR(30)/ATR(90) < 0.8 for low-volatility breakouts), 
provide high-probability entries with minimal whipsaw. Uses discrete position sizing (0.25) and 
ATR(20) stoploss/take-profit to control risk. Targets 12-37 trades/year on 12h timeframe (50-150 
total over 4 years) to minimize fee drag while capturing sustained moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_vol_regime_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for volume spike and ATR regime (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d volume ratio (current vs 20-period average)
    if len(df_1d) >= 20:
        vol_1d = df_1d['volume'].values
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1d = np.zeros(len(vol_1d))
        vol_ratio_1d[20:] = vol_1d[20:] / vol_ma_20[20:]
        vol_ratio_1d[:20] = 1.0  # Neutral for warmup
        vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    else:
        vol_ratio_1d_aligned = np.full(n, 1.0)
    
    # Calculate ATR(30) and ATR(90) on 1d for volatility regime filter
    if len(df_1d) >= 90:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # True Range calculation
        tr = np.zeros(len(close_1d))
        tr[0] = high_1d[0] - low_1d[0]
        for i in range(1, len(close_1d)):
            tr[i] = max(high_1d[i] - low_1d[i], 
                       abs(high_1d[i] - close_1d[i-1]), 
                       abs(low_1d[i] - close_1d[i-1]))
        
        atr_30 = pd.Series(tr).ewm(span=30, min_periods=30, adjust=False).mean().values
        atr_90 = pd.Series(tr).ewm(span=90, min_periods=90, adjust=False).mean().values
        # Low volatility regime: ATR(30) < 0.8 * ATR(90) (compressed volatility)
        vol_regime = atr_30 < (0.8 * atr_90)
        vol_regime_aligned = align_htf_to_ltf(prices, df_1d, vol_regime)
    else:
        vol_regime_aligned = np.full(n, False)
    
    # === 12h Indicators ===
    # Calculate Donchian channel (20-period) on 12h
    lookback = 20
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(n):
        if i >= lookback:
            window_high = high[i-lookback+1:i+1]
            window_low = low[i-lookback+1:i+1]
            donchian_high[i] = np.max(window_high)
            donchian_low[i] = np.min(window_low)
        else:
            donchian_high[i] = np.nan
            donchian_low[i] = np.nan
    
    # Calculate ATR(20) on 12h for stoploss and take-profit
    atr_20 = np.full(n, np.nan)
    if n >= 20:
        tr_12h = np.zeros(n)
        tr_12h[0] = high[0] - low[0]
        for i in range(1, n):
            tr_12h[i] = max(high[i] - low[i], 
                           abs(high[i] - close[i-1]), 
                           abs(low[i] - close[i-1]))
        atr_20_raw = pd.Series(tr_12h).ewm(span=20, min_periods=20, adjust=False).mean().values
        atr_20[:] = atr_20_raw
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    peak_price = 0.0  # For trailing stop in longs
    trough_price = 0.0  # For trailing stop in shorts
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr_20[i]) or np.isnan(vol_ratio_1d_aligned[i]) or
            np.isnan(vol_regime_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Exit Logic (ATR-based stoploss and take-profit) ---
        if in_position:
            atr_val = atr_20[i]
            
            if position_side > 0:  # Long position
                # Update peak price for trailing stop
                if close[i] > peak_price:
                    peak_price = close[i]
                
                # Stoploss: 2.5 * ATR below entry OR 1.5 * ATR below peak (trailing)
                stop_loss = min(entry_price - 2.5 * atr_val, peak_price - 1.5 * atr_val)
                
                # Take profit: 4.0 * ATR above entry
                take_profit = entry_price + 4.0 * atr_val
                
                if low[i] < stop_loss or high[i] > take_profit:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                    
            else:  # Short position
                # Update trough price for trailing stop
                if close[i] < trough_price:
                    trough_price = close[i]
                
                # Stoploss: 2.5 * ATR above entry OR 1.5 * ATR above trough (trailing)
                stop_loss = max(entry_price + 2.5 * atr_val, trough_price + 1.5 * atr_val)
                
                # Take profit: 4.0 * ATR below entry
                take_profit = entry_price - 4.0 * atr_val
                
                if high[i] > stop_loss or low[i] < take_profit:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Breakout conditions: price breaks Donchian channel with volume spike and low volatility regime
        breakout_up = close[i] > donchian_high[i]
        breakout_down = close[i] < donchian_low[i]
        
        volume_confirmation = vol_ratio_1d_aligned[i] > 2.0  # Require strong volume spike
        low_volatility = vol_regime_aligned[i]  # Only trade in compressed volatility regimes
        
        long_entry = breakout_up and volume_confirmation and low_volatility
        short_entry = breakout_down and volume_confirmation and low_volatility
        
        if long_entry:
            in_position = True
            position_side = 1
            entry_price = close[i]
            peak_price = entry_price
            trough_price = entry_price  # Initialize for consistency
            signals[i] = SIZE
        elif short_entry:
            in_position = True
            position_side = -1
            entry_price = close[i]
            trough_price = entry_price
            peak_price = entry_price  # Initialize for consistency
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals