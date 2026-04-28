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
    
    # Get daily data once for HTF context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d Bollinger Bands (20, 2.0)
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    
    # 1d RSI (14)
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_14 = 100 - (100 / (1 + rs))
    rsi_14 = rsi_14.fillna(50).values
    
    # 1d ADX (14)
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    plus_dm = np.concatenate([[np.nan], np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)])
    minus_dm = np.concatenate([[np.nan], np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr_14
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr_14
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align HTF indicators to 6h timeframe
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or 
            np.isnan(rsi_14_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Mean reversion conditions
        near_lower_bb = close[i] <= lower_bb_aligned[i] * 1.005  # Within 0.5% of lower BB
        near_upper_bb = close[i] >= upper_bb_aligned[i] * 0.995  # Within 0.5% of upper BB
        oversold = rsi_14_aligned[i] < 30
        overbought = rsi_14_aligned[i] > 70
        strong_trend = adx_aligned[i] > 25
        
        # Entry conditions - mean reversion in ranging markets, trend following in trending
        # Long: near lower BB + oversold + not strong trend (range) OR oversold + strong trend (trend pullback)
        long_entry = (near_lower_bb and oversold and not strong_trend) or (oversold and strong_trend)
        # Short: near upper BB + overbought + not strong trend (range) OR overbought + strong trend (trend pullback)
        short_entry = (near_upper_bb and overbought and not strong_trend) or (overbought and strong_trend)
        
        # Exit conditions
        long_exit = (close[i] >= sma_20[-1] if not np.isnan(sma_20[-1]) else False) or rsi_14_aligned[i] > 50
        short_exit = (close[i] <= sma_20[-1] if not np.isnan(sma_20[-1]) else False) or rsi_14_aligned[i] < 50
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_BollingerRSI_ADX_MeanReversion_Trend"
timeframe = "6h"
leverage = 1.0