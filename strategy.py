#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_donchian_breakout_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly OHLC
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate Donchian channels (20-period) on weekly data
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Calculate weekly ATR for volatility filter
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1w[0] = tr1[0]
    atr_1w = pd.Series(tr_1w).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate weekly average volume for confirmation
    vol_avg_4w = pd.Series(volume_1w).rolling(window=4, min_periods=4).mean().values
    
    # Align weekly indicators to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    vol_avg_4w_aligned = align_htf_to_ltf(prices, df_1w, vol_avg_4w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from index 40 to ensure sufficient data
    for i in range(40, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(atr_1w_aligned[i]) or np.isnan(vol_avg_4w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Current weekly volume (aligned)
        vol_1w_current = align_htf_to_ltf(prices, df_1w, volume_1w)[i]
        vol_confirm = vol_1w_current > vol_avg_4w_aligned[i]
        
        price = close[i]
        
        # Breakout conditions with volume confirmation
        # Long when price breaks above weekly Donchian high with volume
        long_breakout = (price > donchian_high_aligned[i]) and vol_confirm
        # Short when price breaks below weekly Donchian low with volume
        short_breakout = (price < donchian_low_aligned[i]) and vol_confirm
        
        # Volatility filter: avoid trading in extremely low volatility
        # Only trade when ATR is above 30% of its 20-period average
        atr_ma_20 = pd.Series(atr_1w_aligned).rolling(window=20, min_periods=20).mean()
        atr_ma_20_val = atr_ma_20.iloc[i] if hasattr(atr_ma_20, 'iloc') else atr_ma_20[i] if i < len(atr_ma_20) else np.nan
        vol_filter = not np.isnan(atr_ma_20_val) and atr_1w_aligned[i] > (0.3 * atr_ma_20_val)
        
        if long_breakout and vol_filter and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and vol_filter and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and price < donchian_low_aligned[i]:
            # Exit long when price returns to weekly Donchian low
            position = 0
            signals[i] = 0.0
        elif position == -1 and price > donchian_high_aligned[i]:
            # Exit short when price returns to weekly Donchian high
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals