#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load weekly data ONCE for HTF regime
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    volume_1w = df_1w['volume'].values
    
    # Weekly Donchian channels (20-period)
    upper_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Weekly ATR for volatility filtering
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Weekly RSI for overbought/oversold conditions
    delta = np.diff(close_1w, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14).mean().values
    rs = avg_gain / avg_loss
    rsi_14_1w = 100 - (100 / (1 + rs))
    
    # Align weekly indicators to daily timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_1w, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1w, lower_20)
    atr_14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_14_1w)
    rsi_14_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_14_1w)
    
    # Daily price data
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if NaN in weekly indicators
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or
            np.isnan(atr_14_1w_aligned[i]) or np.isnan(rsi_14_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        upper = upper_20_aligned[i]
        lower = lower_20_aligned[i]
        atr_val = atr_14_1w_aligned[i]
        rsi_val = rsi_14_1w_aligned[i]
        
        # Volume filter: current volume above 20-day average
        vol_ma_20 = np.mean(volume[max(0, i-19):i+1])
        vol_filter = vol > vol_ma_20
        
        # Volatility regime: only trade when volatility is below 70th percentile
        vol_regime = atr_val < np.nanpercentile(atr_14_1w_aligned[:i+1], 70)
        
        # RSI filter: avoid extreme overbought/oversold conditions
        rsi_filter = (rsi_val > 30) & (rsi_val < 70)
        
        if position == 0:
            # Long: price breaks above weekly upper Donchian with volume and regime filters
            if price > upper and vol_filter and vol_regime and rsi_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly lower Donchian with volume and regime filters
            elif price < lower and vol_filter and vol_regime and rsi_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below weekly lower Donchian or volatility spikes
            if price < lower or atr_val > np.nanpercentile(atr_14_1w_aligned[:i+1], 85):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above weekly upper Donchian or volatility spikes
            if price > upper or atr_val > np.nanpercentile(atr_14_1w_aligned[:i+1], 85):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyDonchianBreakout_VolumeVolatilityRSIFilter"
timeframe = "1d"
leverage = 1.0