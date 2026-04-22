#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # 1d data for higher timeframe trend and volatility
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d ATR(14) for volatility filter
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(np.abs(low_1d[1:] - close_1d[:-1]), np.abs(high_1d[:-1] - low_1d[1:]))
    tr = np.concatenate([[np.nan], np.maximum(tr1, tr2)])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # 1d ADX(14) for trend strength
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr_abs = np.where(np.isnan(tr), 0, tr)
    atr_14_smooth = pd.Series(tr_abs).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_14_smooth
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_14_smooth
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 4h RSI(14) for momentum
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 4h Bollinger Bands(20,2) for volatility and mean reversion
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_band = sma20 + 2 * std20
    lower_band = sma20 - 2 * std20
    
    # Volume confirmation
    vol_ma20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_surge = prices['volume'].values > 1.8 * vol_ma20
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(atr_14_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(sma20[i]) or np.isnan(std20[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        atr14d = atr_14_aligned[i]
        adx14d = adx_aligned[i]
        rsi14 = rsi[i]
        bb_mid = sma20[i]
        bb_width = (upper_band[i] - lower_band[i]) / bb_mid if bb_mid != 0 else 0
        
        if position == 0:
            # Long: Low volatility (BB width < 0.04) + strong trend (ADX > 25) + RSI < 30 + volume surge
            if (bb_width < 0.04 and adx14d > 25 and rsi14 < 30 and 
                close[i] > bb_mid and vol_surge[i]):
                signals[i] = 0.25
                position = 1
            # Short: Low volatility + strong trend + RSI > 70 + volume surge
            elif (bb_width < 0.04 and adx14d > 25 and rsi14 > 70 and 
                  close[i] < bb_mid and vol_surge[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: volatility expansion or trend weakening
            if position == 1:
                if bb_width > 0.06 or adx14d < 20 or rsi14 > 70:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if bb_width > 0.06 or adx14d < 20 or rsi14 < 30:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_VolatilityTrend_RSI_VolumeSurge_v1"
timeframe = "4h"
leverage = 1.0