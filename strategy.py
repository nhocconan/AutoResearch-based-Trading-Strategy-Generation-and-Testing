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
    
    # Get 1d data once for HTF context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d Williams %R (14) - momentum oscillator
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid div by zero
    
    # 1d RSI (14) - momentum confirmation
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # 1d EMA21 - trend filter
    ema_21_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # 1d ATR14 - volatility filter
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align HTF indicators to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    ema_21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Volume ratio for confirmation (20-period MA)
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume_1d / vol_ma_1d
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(ema_21_1d_aligned[i]) or np.isnan(atr_14_aligned[i]) or
            np.isnan(vol_ratio_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Williams %R conditions: oversold (< -80) for long, overbought (> -20) for short
        wr_oversold = williams_r_aligned[i] < -80
        wr_overbought = williams_r_aligned[i] > -20
        
        # RSI conditions: not extreme, momentum confirmation
        rsi_not_overbought = rsi_aligned[i] < 70
        rsi_not_oversold = rsi_aligned[i] > 30
        rsi_bullish = rsi_aligned[i] > 50
        rsi_bearish = rsi_aligned[i] < 50
        
        # Trend filter: price above/below 1d EMA21
        trend_up = close[i] > ema_21_1d_aligned[i]
        trend_down = close[i] < ema_21_1d_aligned[i]
        
        # Volatility filter: avoid extremely low volatility periods
        vol_filter = atr_14_aligned[i] > 0.005 * close[i]  # ATR > 0.5% of price
        
        # Volume confirmation: require above-average volume
        vol_confirm = vol_ratio_aligned[i] > 1.3
        
        # Entry conditions - Williams %R mean reversion with trend filter
        # Long: oversold Williams + bullish RSI + uptrend + vol filter + volume confirmation
        long_entry = wr_oversold and rsi_bullish and trend_up and vol_filter and vol_confirm
        # Short: overbought Williams + bearish RSI + downtrend + vol filter + volume confirmation
        short_entry = wr_overbought and rsi_bearish and trend_down and vol_filter and vol_confirm
        
        # Exit conditions: Williams %R reverses or trend changes
        long_exit = (williams_r_aligned[i] > -50) or (not trend_up)
        short_exit = (williams_r_aligned[i] < -50) or (not trend_down)
        
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

name = "6h_WilliamsRSI_MeanReversion_1dEMA21_Trend"
timeframe = "6h"
leverage = 1.0