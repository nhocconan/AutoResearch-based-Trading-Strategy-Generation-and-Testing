#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wEMA34_Trend_VolumeSpike_v1
Hypothesis: On daily timeframe, price breaking above Camarilla R1 or below S1 levels from prior daily session captures institutional breakouts. Combined with 1-week EMA34 trend filter, volume spike confirmation, and ATR-based stoploss. Designed for low trade frequency (~15-25/year) to minimize fee drag and work in both bull (breakout continuation) and bear (breakdown continuation) regimes by following the weekly trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1-week for EMA trend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # === 1-week EMA34 for trend filter ===
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === Daily Camarilla levels from prior daily session (HLC of previous day) ===
    # Note: For daily timeframe, we use the previous day's OHLC to calculate today's Camarilla levels
    high_1d = df_1w['high'].values  # This is actually weekly high, need to get daily data differently
    # Correction: For 1d timeframe, we need to load daily data for Camarilla calculation
    # But since we're on 1d timeframe, prices already contain daily data
    # So we need to calculate Camarilla from previous day's values within the prices dataframe
    
    # For 1d timeframe, calculate Camarilla levels using prior day's OHLC from prices
    high_prices = prices['high'].values
    low_prices = prices['low'].values
    close_prices = prices['close'].values
    
    # Shift by 1 to get previous day's values
    prev_high = np.roll(high_prices, 1)
    prev_low = np.roll(low_prices, 1)
    prev_close = np.roll(close_prices, 1)
    # First bar has no previous day
    prev_high[0] = high_prices[0]
    prev_low[0] = low_prices[0]
    prev_close[0] = close_prices[0]
    
    # Camarilla R1, S1, R2, S2 from previous day
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    camarilla_r2 = prev_close + (prev_high - prev_low) * 1.1 / 6
    camarilla_s2 = prev_close - (prev_high - prev_low) * 1.1 / 6
    
    # === Volume spike filter (20-period) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    # === ATR for stoploss (14-period) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        ema_34 = ema_34_1w_aligned[i]
        vol_spike = vol_ratio[i]
        r1 = camarilla_r1[i]
        s1 = camarilla_s1[i]
        r2 = camarilla_r2[i]
        s2 = camarilla_s2[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: price breaks above R1 (bullish breakout) + above weekly EMA34 + volume spike > 1.5
            if price_close > r1 and price_close > ema_34 and vol_spike > 1.5:
                signals[i] = 0.25
                position = 1
                entry_price = price_close
            # Short: price breaks below S1 (bearish breakdown) + below weekly EMA34 + volume spike > 1.5
            elif price_close < s1 and price_close < ema_34 and vol_spike > 1.5:
                signals[i] = -0.25
                position = -1
                entry_price = price_close
        
        elif position != 0:
            # Stoploss: 2 * ATR from entry
            if position == 1:
                if price_close < entry_price - 2.0 * atr_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price_close > entry_price + 2.0 * atr_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wEMA34_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0