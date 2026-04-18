#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for calculations (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly RSI (14-period)
    delta_w = np.diff(df_1w['close'], prepend=df_1w['close'][0])
    gain_w = np.where(delta_w > 0, delta_w, 0)
    loss_w = np.where(delta_w < 0, -delta_w, 0)
    avg_gain_w = pd.Series(gain_w).rolling(window=14, min_periods=14).mean().values
    avg_loss_w = pd.Series(loss_w).rolling(window=14, min_periods=14).mean().values
    rs_w = avg_gain_w / (avg_loss_w + 1e-10)
    rsi_1w = 100 - (100 / (1 + rs_w))
    
    # Calculate weekly ATR (14-period) for volatility filter
    tr1_w = df_1w['high'] - df_1w['low']
    tr2_w = np.abs(df_1w['high'] - np.roll(df_1w['close'], 1))
    tr3_w = np.abs(df_1w['low'] - np.roll(df_1w['close'], 1))
    tr2_w[0] = np.nan
    tr3_w[0] = np.nan
    tr_w = np.maximum(tr1_w, np.maximum(tr2_w, tr3_w))
    atr_1w = pd.Series(tr_w).rolling(window=14, min_periods=14).mean().values
    
    # Calculate weekly SMA (20-period) for trend filter
    sma_20w = pd.Series(df_1w['close']).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to daily timeframe
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    sma_20w_aligned = align_htf_to_ltf(prices, df_1w, sma_20w)
    
    # Calculate daily ATR (14-period) for position sizing
    tr1_d = high - low
    tr2_d = np.abs(high - np.roll(close, 1))
    tr3_d = np.abs(low - np.roll(close, 1))
    tr2_d[0] = np.nan
    tr3_d[0] = np.nan
    tr_d = np.maximum(tr1_d, np.maximum(tr2_d, tr3_d))
    atr_14d = pd.Series(tr_d).rolling(window=14, min_periods=14).mean().values
    
    # Calculate daily volume MA (20-period) for volume filter
    vol_ma_20d = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Wait for enough data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi_1w_aligned[i]) or
            np.isnan(atr_1w_aligned[i]) or
            np.isnan(sma_20w_aligned[i]) or
            np.isnan(atr_14d[i]) or
            np.isnan(vol_ma_20d[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: weekly ATR > 20-period average
        if i >= 20:
            atr_ma_20w = pd.Series(atr_1w).rolling(window=20, min_periods=20).mean().values
            atr_ma_20w_aligned = align_htf_to_ltf(prices, df_1w, atr_ma_20w)
            vol_filter = not np.isnan(atr_ma_20w_aligned[i]) and atr_1w_aligned[i] > atr_ma_20w_aligned[i]
        else:
            vol_filter = False
        
        # Trend filter: price above/below weekly SMA20
        price_above_sma20w = close[i] > sma_20w_aligned[i]
        price_below_sma20w = close[i] < sma_20w_aligned[i]
        
        # Volume filter: current volume > 20-day average
        vol_filter_daily = volume[i] > vol_ma_20d[i]
        
        trade_allowed = vol_filter and price_above_sma20w and vol_filter_daily
        trade_allowed_short = vol_filter and price_below_sma20w and vol_filter_daily
        
        if position == 0:
            # Long: weekly RSI < 40 (not overbought) and price above weekly SMA20
            if trade_allowed and rsi_1w_aligned[i] < 40:
                signals[i] = 0.25
                position = 1
            # Short: weekly RSI > 60 (not oversold) and price below weekly SMA20
            elif trade_allowed_short and rsi_1w_aligned[i] > 60:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: weekly RSI > 50 or price below weekly SMA20
            if rsi_1w_aligned[i] > 50 or close[i] < sma_20w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: weekly RSI < 50 or price above weekly SMA20
            if rsi_1w_aligned[i] < 50 or close[i] > sma_20w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyRSI_SMA20_VolumeFilter_v1"
timeframe = "1d"
leverage = 1.0