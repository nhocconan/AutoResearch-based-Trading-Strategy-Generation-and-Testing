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
    
    # Get weekly data for higher timeframe context (as per experiment spec)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate weekly ATR(14) for volatility filter
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_14_1w)
    
    # Calculate weekly EMA(34) for trend direction
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate weekly RSI(14) for overbought/oversold
    delta_1w = pd.Series(close_1w).diff()
    gain_1w = delta_1w.where(delta_1w > 0, 0)
    loss_1w = -delta_1w.where(delta_1w < 0, 0)
    avg_gain_1w = gain_1w.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss_1w = loss_1w.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs_1w = avg_gain_1w / avg_loss_1w
    rsi_1w = 100 - (100 / (1 + rs_1w))
    rsi_1w = rsi_1w.values
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Calculate weekly volume moving average
    vol_ma_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(rsi_1w_aligned[i]) or 
            np.isnan(atr_14_1w_aligned[i]) or 
            np.isnan(vol_ma_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below weekly EMA34
        price_above_ema = close[i] > ema_34_1w_aligned[i]
        price_below_ema = close[i] < ema_34_1w_aligned[i]
        
        # Volatility filter: ATR below median (avoid high volatility periods)
        atr_median = np.nanmedian(atr_14_1w_aligned[:i+1]) if i >= 50 else atr_14_1w_aligned[i]
        low_volatility = atr_14_1w_aligned[i] < atr_median
        
        # RSI filter: avoid extreme overbought/oversold conditions
        rsi_not_overbought = rsi_1w_aligned[i] < 70
        rsi_not_oversold = rsi_1w_aligned[i] > 30
        
        # Volume filter: current volume above weekly average
        volume_filter = volume[i] > vol_ma_1w_aligned[i]
        
        # Long conditions: price above weekly EMA34 + low volatility + RSI not overbought + volume
        long_condition = (price_above_ema and 
                         low_volatility and 
                         rsi_not_overbought and 
                         volume_filter)
        
        # Short conditions: price below weekly EMA34 + low volatility + RSI not oversold + volume
        short_condition = (price_below_ema and 
                          low_volatility and 
                          rsi_not_oversold and 
                          volume_filter)
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: trend reversal or volatility spike
        elif position == 1 and (not price_above_ema or not low_volatility):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (not price_below_ema or not low_volatility):
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_EMA34_RSI_VolumeFilter_1wTrend"
timeframe = "1d"
leverage = 1.0