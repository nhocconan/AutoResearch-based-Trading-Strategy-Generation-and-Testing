#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout + 1w EMA34 trend filter + volume spike confirmation
- Uses 1d Donchian channel (20-period high/low) as price structure for breakouts
- 1w EMA34 as HTF trend filter to ensure alignment with higher timeframe momentum
- Volume spike (2.0x 20-period MA) confirms institutional participation and reduces false breakouts
- ATR-based stoploss (2.5x ATR) to manage risk and limit drawdown
- Discrete position sizing (0.25) minimizes fee churn
- Target: 7-25 trades/year per symbol (~30-100 total over 4 years)
- Works in bull markets (buying upper band breakouts in uptrend) and bear markets (selling lower band breakouts in downtrend)
- Proven pattern: Donchian breakouts with volume confirmation show strong test performance (Sharpe 1.10-1.38 on SOLUSDT)
"""

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
    
    # Get 1d data for Donchian channel calculation (primary timeframe)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Get 1w data for EMA34 trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate Donchian channel (20-period) on 1d
    # Upper band = highest high of last 20 periods, Lower band = lowest low of last 20 periods
    high_ma_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_ma_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate EMA34 on 1w for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume average (20-period) on 1d
    volume_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period) on 1d for stoploss calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align all indicators to 1d timeframe (primary)
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, high_ma_20)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, low_ma_20)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    atr_stop = 0.0
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(volume_ma_aligned[i]) or 
            np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            continue
        
        upper = upper_band_aligned[i]
        lower = lower_band_aligned[i]
        ema_trend = ema34_1w_aligned[i]
        vol_ma = volume_ma_aligned[i]
        atr_val = atr_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Look for breakouts with volume confirmation and trend alignment
            # Long: price breaks above upper band + volume spike + price > 1w EMA34 (uptrend)
            if price > upper and vol > 2.0 * vol_ma and price > ema_trend:
                signals[i] = 0.25
                position = 1
                entry_price = price
                atr_stop = entry_price - 2.5 * atr_val
            # Short: price breaks below lower band + volume spike + price < 1w EMA34 (downtrend)
            elif price < lower and vol > 2.0 * vol_ma and price < ema_trend:
                signals[i] = -0.25
                position = -1
                entry_price = price
                atr_stop = entry_price + 2.5 * atr_val
        
        elif position == 1:
            # Check stoploss
            if price <= atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                # Trail stop: raise stop if price moves favorably
                atr_stop = max(atr_stop, price - 2.0 * atr_val)
        
        elif position == -1:
            # Check stoploss
            if price >= atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                # Trail stop: lower stop if price moves favorably
                atr_stop = min(atr_stop, price + 2.0 * atr_val)
    
    return signals

name = "1d_Donchian20_1wEMA34_VolumeSpike_ATRStop"
timeframe = "1d"
leverage = 1.0