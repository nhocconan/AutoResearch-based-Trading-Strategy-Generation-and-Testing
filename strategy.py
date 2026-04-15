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
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d ATR(14) for volatility filter
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr3 = np.abs(df_1d['low'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 1d RSI(14) for mean reversion
    delta = pd.Series(df_1d['close'].values).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14_1d = 100 - (100 / (1 + rs))
    rsi_14_1d_values = rsi_14_1d.values
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d_values)
    
    # Calculate 1d ADX(14) for trend strength filter
    plus_dm = np.where((df_1d['high'].diff()) > (df_1d['low'].diff().abs()), df_1d['high'].diff(), 0)
    minus_dm = np.where((df_1d['low'].diff().abs()) > (df_1d['high'].diff()), df_1d['low'].diff().abs(), 0)
    plus_dm = np.where(plus_dm < 0, 0, plus_dm)
    minus_dm = np.where(minus_dm < 0, 0, minus_dm)
    tr = np.maximum(df_1d['high'] - df_1d['low'], np.maximum(np.abs(df_1d['high'] - np.roll(df_1d['close'], 1)), np.abs(df_1d['low'] - np.roll(df_1d['close'], 1))))
    tr[0] = df_1d['high'].iloc[0] - df_1d['low'].iloc[0]
    atr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean() / (atr_14 + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean() / (atr_14 + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_14 = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(rsi_14_1d_aligned[i]) or np.isnan(adx_14_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 1d EMA50
        trend_filter = close[i] > ema_50_1d_aligned[i]
        
        # Volatility filter: only trade when volatility is elevated
        vol_filter = atr_14_1d_aligned[i] > 0.003 * close[i]
        
        # Trend strength filter: only trade when ADX > 20
        trend_strength = adx_14_aligned[i] > 20
        
        # Mean reversion conditions using RSI extremes
        # Long: RSI < 30 (oversold) + bullish trend + volatility
        # Short: RSI > 70 (overbought) + bearish trend + volatility
        if (not trend_filter and  # bearish bias
            rsi_14_1d_aligned[i] < 30 and  # oversold
            vol_filter and
            trend_strength):
            signals[i] = 0.25
        elif (trend_filter and  # bullish bias
              rsi_14_1d_aligned[i] > 70 and  # overbought
              vol_filter and
              trend_strength):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_RSI_MeanReversion_TrendFilter_v1"
timeframe = "1d"
leverage = 1.0