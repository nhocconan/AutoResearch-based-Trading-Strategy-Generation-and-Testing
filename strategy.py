#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for volatility regime and trend
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily ATR(14) for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[high_1d[0] - low_1d[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = np.zeros(len(df_1d))
    for i in range(len(df_1d)):
        if i < 14:
            atr_1d[i] = np.mean(tr_1d[:i+1]) if i > 0 else tr_1d[0]
        else:
            atr_1d[i] = (atr_1d[i-1] * 13 + tr_1d[i]) / 14
    
    # Daily EMA(50) for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily indicators to 1h
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1h Bollinger Bands (20, 2.0)
    close_series = pd.Series(close)
    bb_mid = close_series.rolling(window=20, min_periods=20).mean()
    bb_std = close_series.rolling(window=20, min_periods=20).std()
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std
    
    # Bollinger Band Width for squeeze detection
    bb_width = (bb_upper - bb_lower) / bb_mid
    
    # 1h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    for i in range(n):
        if i < 14:
            avg_gain[i] = np.mean(gain[:i+1]) if i > 0 else gain[0]
            avg_loss[i] = np.mean(loss[:i+1]) if i > 0 else loss[0]
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike detection (2.0x 20-period average)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0
    
    # Session filter: 08-20 UTC (avoid low liquidity periods)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    start_idx = max(50, 20)  # Need enough history for indicators
    
    for i in range(start_idx, n):
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
            
        if (np.isnan(atr_1d_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(bb_width[i]) or
            np.isnan(rsi[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 0 else 0
        volume_spike = vol_ratio > 2.0
        
        # Bollinger Band squeeze: BB width below 20-period mean
        bb_width_ma = np.mean(bb_width[max(0, i-20):i+1]) if not np.isnan(np.mean(bb_width[max(0, i-20):i+1])) else 0
        bb_squeeze = bb_width[i] < bb_width_ma * 0.8
        
        # Trend filter: price above/below daily EMA(50)
        price_above_ema = price > ema_50_1d_aligned[i]
        price_below_ema = price < ema_50_1d_aligned[i]
        
        # Volatility filter: avoid extremely low volatility
        vol_filter = atr_1d_aligned[i] > np.mean(atr_1d_aligned[max(0, i-20):i+1]) * 0.5
        
        if position == 0:
            # Long: BB squeeze breakout above upper band with volume and trend
            if bb_squeeze and volume_spike and vol_filter and price_above_ema and price > bb_upper[i]:
                signals[i] = 0.20
                position = 1
            # Short: BB squeeze breakout below lower band with volume and trend
            elif bb_squeeze and volume_spike and vol_filter and price_below_ema and price < bb_lower[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price returns to middle band or volatility drops
            if price < bb_mid[i] or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price returns to middle band or volatility drops
            if price > bb_mid[i] or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_BollingerSqueeze_Breakout_VolumeTrend"
timeframe = "1h"
leverage = 1.0