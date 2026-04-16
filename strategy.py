# Hypothesis: A 12-hour strategy combining weekly Donchian breakouts (trend), daily EMA trend filter (trend alignment), and volume confirmation (momentum) to capture medium-term trends across bull and bear markets. The weekly timeframe provides robust trend context, while daily EMA and volume filters ensure entries align with intermediate momentum. Designed for low trade frequency (<30/year) to minimize fee drag, with volatility-adjusted position sizing via ATR-based stops. Focus on BTC/ETH as primary assets.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Weekly data (HTF trend filter) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # === Daily data (trend and volume filter) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === Weekly Donchian channel (20-period) ===
    donch_high_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donch_low_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to avoid look-ahead (use previous bar's channel)
    donch_high_1w = np.roll(donch_high_1w, 1)
    donch_low_1w = np.roll(donch_low_1w, 1)
    donch_high_1w[0] = np.nan
    donch_low_1w[0] = np.nan
    
    # === Daily EMA34 (trend filter) ===
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === Daily volume ratio for confirmation ===
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = volume_1d / vol_ma_20_1d
    
    # === 14-period ATR for stoploss (using 12h data) ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(donch_high_1w[i]) or 
            np.isnan(donch_low_1w[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ratio_1d[i]) or
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        upper = donch_high_1w[i]
        lower = donch_low_1w[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_ratio = vol_ratio_1d[i]
        atr = atr_14[i]
        
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
            # Exit: price closes below Donchian lower OR trend reverses (below daily EMA34)
            if price < lower or price < ema_trend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper OR trend reverses (above daily EMA34)
            if price > upper or price > ema_trend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Break above weekly Donchian upper with volume, in uptrend (above daily EMA34)
            if price > upper and vol_ratio > 2.0 and price > ema_trend:
                signals[i] = 0.25
                position = 1
                entry_price = price
                continue
            # SHORT: Break below weekly Donchian lower with volume, in downtrend (below daily EMA34)
            elif price < lower and vol_ratio > 2.0 and price < ema_trend:
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

name = "12h_WeeklyDonchian_DailyEMA34_Volume_ATRStop"
timeframe = "12h"
leverage = 1.0