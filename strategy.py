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
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for volatility regime filter
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr3 = np.abs(df_1d['low'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d Donchian(20) channels
    donchian_high_20 = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    
    # Calculate 1d volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    volume_ratio = volume / (vol_ma_20_aligned + 1e-10)
    
    # Calculate 1d RSI(14) for mean reversion filter
    delta = pd.Series(df_1d['close'].values).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14 = 100 - (100 / (1 + rs))
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_14_1d_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(donchian_high_20_aligned[i]) or np.isnan(donchian_low_20_aligned[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(rsi_14_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility regime filter: only trade when daily ATR is elevated (> 0.6% of price)
        vol_regime = atr_14_1d_aligned[i] > 0.006 * close[i]
        
        # Trend filter: price relative to daily EMA50
        trend_filter = close[i] > ema_50_1d_aligned[i]
        
        # Mean reversion filter: avoid extreme RSI
        rsi_filter = (rsi_14_aligned[i] > 30) & (rsi_14_aligned[i] < 70)
        
        # Long conditions:
        # 1. Price above daily EMA50 (bullish bias)
        # 2. Price breaks above daily Donchian(20) high with volume (bullish breakout)
        # 3. Volume confirmation: volume > 2.0x average (stricter)
        # 4. Daily volatility regime filter
        # 5. RSI not overbought/oversold
        if (trend_filter and
            close[i] > donchian_high_20_aligned[i] and
            volume_ratio[i] > 2.0 and
            vol_regime and
            rsi_filter):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price below daily EMA50 (bearish bias)
        # 2. Price breaks below daily Donchian(20) low with volume (bearish breakdown)
        # 3. Volume confirmation: volume > 2.0x average
        # 4. Daily volatility regime filter
        # 5. RSI not overbought/oversold
        elif (not trend_filter and
              close[i] < donchian_low_20_aligned[i] and
              volume_ratio[i] > 2.0 and
              vol_regime and
              rsi_filter):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_Vol_Regime_Donchian20_1dEMA50_RSI_Filter_v1"
timeframe = "4h"
leverage = 1.0