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
    
    # Calculate 1d EMA34 for trend filter (needs completed 1d candle, no extra delay)
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d RSI(14) for mean reversion signals in ranging markets
    delta = pd.Series(df_1d['close'].values).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14_1d = 100 - (100 / (1 + rs))
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # Calculate 12h ATR(14) for volatility filter and position sizing
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_12h = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 12h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(rsi_14_1d_aligned[i]) or 
            np.isnan(atr_14_12h[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: ranging market when RSI between 40-60 (avoid strong trends)
        ranging_market = (rsi_14_1d_aligned[i] >= 40) & (rsi_14_1d_aligned[i] <= 60)
        
        # Volatility filter: sufficient volatility for meaningful moves
        vol_filter = atr_14_12h[i] > 0.003 * close[i]
        
        # Volume confirmation: above average participation
        vol_confirm = volume_ratio[i] > 1.2
        
        # Long conditions: mean reversion in ranging market
        # Price below EMA34 (slightly bearish short-term) + oversold RSI + ranging + vol
        if (close[i] < ema_34_1d_aligned[i] and
            rsi_14_1d_aligned[i] < 35 and
            ranging_market and
            vol_filter and
            vol_confirm):
            signals[i] = 0.25
            
        # Short conditions: mean reversion in ranging market
        # Price above EMA34 (slightly bullish short-term) + overbought RSI + ranging + vol
        elif (close[i] > ema_34_1d_aligned[i] and
              rsi_14_1d_aligned[i] > 65 and
              ranging_market and
              vol_filter and
              vol_confirm):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_EMA34_RSI_Volume_Regime_Filter_v1"
timeframe = "12h"
leverage = 1.0