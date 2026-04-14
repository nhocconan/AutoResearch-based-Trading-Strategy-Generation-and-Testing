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
    
    # Load weekly data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    volume_1w = df_1w['volume'].values
    
    # Weekly ATR for volatility measurement (14-period)
    high_low = high_1w - low_1w
    high_close = np.abs(high_1w - np.roll(close_1w, 1))
    low_close = np.abs(low_1w - np.roll(close_1w, 1))
    high_close[0] = high_low[0]
    low_close[0] = high_low[0]
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    tr_series = pd.Series(tr)
    atr_14w = tr_series.rolling(window=14, min_periods=14).mean().values
    
    # Weekly EMA for trend direction (21-period)
    close_1w_series = pd.Series(close_1w)
    ema_21w = close_1w_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Weekly volume average (20-period)
    vol_1w_series = pd.Series(volume_1w)
    vol_ma_20w = vol_1w_series.rolling(window=20, min_periods=20).mean().values
    
    # Align weekly indicators to daily timeframe
    ema_21w_aligned = align_htf_to_ltf(prices, df_1w, ema_21w)
    atr_14w_aligned = align_htf_to_ltf(prices, df_1w, atr_14w)
    vol_ma_20w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20w)
    
    # Daily indicators
    # Daily RSI (14-period) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_series = pd.Series(gain)
    loss_series = pd.Series(loss)
    avg_gain = gain_series.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = loss_series.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Daily volume average (20-period)
    vol_series = pd.Series(volume)
    vol_ma_20d = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_21w_aligned[i]) or np.isnan(atr_14w_aligned[i]) or 
            np.isnan(vol_ma_20w_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma_20d[i])):
            continue
        
        # Trend filter: price above/below weekly EMA
        price_above_ema = close[i] > ema_21w_aligned[i]
        price_below_ema = close[i] < ema_21w_aligned[i]
        
        # Momentum filter: RSI not extreme
        rsi_not_overbought = rsi[i] < 70
        rsi_not_oversold = rsi[i] > 30
        
        # Volume filter: current volume > 1.5x weekly average
        volume_filter = volume[i] > vol_ma_20w_aligned[i] * 1.5
        
        if position == 0:
            # Long: Uptrend + moderate RSI + volume surge
            if (price_above_ema and rsi_not_overbought and volume_filter):
                position = 1
                signals[i] = position_size
            # Short: Downtrend + moderate RSI + volume surge
            elif (price_below_ema and rsi_not_oversold and volume_filter):
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: trend reversal or RSI overbought
            if (not price_above_ema) or rsi[i] >= 70:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: trend reversal or RSI oversold
            if (not price_below_ema) or rsi[i] <= 30:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "1d_1w_EMA21_RSI_VolumeSurge_TrendFollow_v1"
timeframe = "1d"
leverage = 1.0