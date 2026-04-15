#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Donchian channel breakouts with volume confirmation
# and ATR-based position sizing. Weekly structure provides strong trend filter for BTC/ETH
# in both bull and bear markets, while daily timeframe allows timely entries. Low trade
# frequency (<25/year) minimizes fee drag. Uses discrete position sizes (0.0, ±0.25) to
# reduce churn. ATR stoploss manages risk during volatile periods.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period)
    highest_high_20 = pd.Series(df_1w['high'].values).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(df_1w['low'].values).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, highest_high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, lowest_low_20)
    
    # Calculate weekly ATR(14) for volatility filter and position sizing
    tr1 = df_1w['high'] - df_1w['low']
    tr2 = np.abs(df_1w['high'] - np.concatenate([[df_1w['close'].iloc[0]], df_1w['close'].iloc[:-1]]))
    tr3 = np.abs(df_1w['low'] - np.concatenate([[df_1w['close'].iloc[0]], df_1w['close'].iloc[:-1]]))
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1w = pd.Series(tr_1w).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_14_1w)
    
    # Calculate daily volume average (20-period) for volume confirmation
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(atr_14_1w_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-day average
        vol_confirm = volume[i] > 1.5 * volume_ma_20[i]
        
        # Volatility filter: only trade when weekly ATR is reasonable (< 8% of price)
        vol_filter = atr_14_1w_aligned[i] < 0.08 * close[i]
        
        # Long breakout: price breaks above weekly Donchian high
        if (close[i] > donchian_high_aligned[i] and vol_confirm and vol_filter):
            signals[i] = 0.25
            
        # Short breakout: price breaks below weekly Donchian low
        elif (close[i] < donchian_low_aligned[i] and vol_confirm and vol_filter):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_WeeklyDonchian20_Volume_Breakout_v1"
timeframe = "1d"
leverage = 1.0