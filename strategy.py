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
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for volatility regime filter
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr3 = np.abs(df_1d['low'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 1d RSI(14) for mean reversion filter
    delta = df_1d['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi_14_1d = 100 - (100 / (1 + rs))
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d.values)
    
    # Calculate 1d Donchian channels (20-period)
    donchian_high_20 = df_1d['high'].rolling(window=20, min_periods=20).max().values
    donchian_low_20 = df_1d['low'].rolling(window=20, min_periods=20).min().values
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_14_1d_aligned[i]) or np.isnan(rsi_14_1d_aligned[i]) or 
            np.isnan(donchian_high_20_aligned[i]) or np.isnan(donchian_low_20_aligned[i]) or 
            np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Volatility regime filter: only trade when daily ATR is elevated (> 0.5% of price)
        vol_regime = atr_14_1d_aligned[i] > 0.005 * close[i]
        
        # Long conditions:
        # 1. Price above 1d Donchian mid-line (bullish bias)
        # 2. Price breaks above 1d Donchian upper band with volume (breakout continuation)
        # 3. Volume confirmation: volume > 1.3x average
        # 4. Daily volatility regime filter (avoid chop)
        # 5. RSI not overbought (< 70) to avoid exhaustion
        mid_line = (donchian_high_20_aligned[i] + donchian_low_20_aligned[i]) / 2.0
        if (close[i] > mid_line and
            close[i] > donchian_high_20_aligned[i] and
            volume_ratio[i] > 1.3 and
            vol_regime and
            rsi_14_1d_aligned[i] < 70):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price below 1d Donchian mid-line (bearish bias)
        # 2. Price breaks below 1d Donchian lower band with volume (breakdown continuation)
        # 3. Volume confirmation: volume > 1.3x average
        # 4. Daily volatility regime filter
        # 5. RSI not oversold (> 30) to avoid exhaustion
        elif (close[i] < mid_line and
              close[i] < donchian_low_20_aligned[i] and
              volume_ratio[i] > 1.3 and
              vol_regime and
              rsi_14_1d_aligned[i] > 30):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_Donchian20_Volume_RSI_Filter_v1"
timeframe = "1d"
leverage = 1.0