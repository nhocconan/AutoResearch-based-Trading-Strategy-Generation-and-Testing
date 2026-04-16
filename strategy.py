# 1d_Donchian_1wEMA34_Volume_ATRStop_v1
# Hypothesis: 1-day timeframe with weekly EMA trend filter, Donchian breakouts, volume confirmation, and ATR-based stops.
# Weekly EMA34 provides robust trend filtering for both bull and bear markets, reducing whipsaws.
# Donchian breakouts capture momentum, volume confirms strength, and ATR stops manage risk.
# Target: 10-30 trades/year to minimize fee drag while capturing major trends.
# Uses 1d primary and 1w HTF as specified in experiment parameters.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Weekly data (HTF for trend filter) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # === Weekly EMA34 (trend filter) ===
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === Daily Donchian channel (20-period) ===
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to avoid look-ahead (use previous bar's channel)
    donch_high = np.roll(donch_high, 1)
    donch_low = np.roll(donch_low, 1)
    donch_high[0] = np.nan
    donch_low[0] = np.nan
    
    # === Daily volume ratio for confirmation ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma_20
    
    # === 10-period ATR for stoploss ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0
    atr_10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(vol_ratio[i]) or
            np.isnan(atr_10[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        upper = donch_high[i]
        lower = donch_low[i]
        ema_trend = ema_34_1w_aligned[i]
        vol_ratio_val = vol_ratio[i]
        atr = atr_10[i]
        
        # === STOPLOSS LOGIC ===
        if position == 1:  # Long position
            # Stop loss: price closes below entry - 2.5 * ATR
            if price < entry_price - 2.5 * atr:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Stop loss: price closes above entry + 2.5 * ATR
            if price > entry_price + 2.5 * atr:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower OR trend reverses (below weekly EMA34)
            if price < lower or price < ema_trend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper OR trend reverses (above weekly EMA34)
            if price > upper or price > ema_trend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Break above Donchian upper with volume, in uptrend (above weekly EMA34)
            if price > upper and vol_ratio_val > 2.0 and price > ema_trend:
                signals[i] = 0.25
                position = 1
                entry_price = price
                continue
            # SHORT: Break below Donchian lower with volume, in downtrend (below weekly EMA34)
            elif price < lower and vol_ratio_val > 2.0 and price < ema_trend:
                signals[i] = -0.25
                position = -1
                entry_price = price
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_Donchian_1wEMA34_Volume_ATRStop_v1"
timeframe = "1d"
leverage = 1.0