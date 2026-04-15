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
    
    # Get 1w HTF data once before loop for primary trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA(34) for long-term trend
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Get 1d HTF data for volatility and regime filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily ATR(14) for volatility filter
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr3 = np.abs(df_1d['low'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate daily ADX(14) for trend strength regime filter
    plus_dm = np.where((df_1d['high'] - df_1d['high'].shift(1)) > (df_1d['low'].shift(1) - df_1d['low']), 
                       np.maximum(df_1d['high'] - df_1d['high'].shift(1), 0), 0)
    minus_dm = np.where((df_1d['low'].shift(1) - df_1d['low']) > (df_1d['high'] - df_1d['high'].shift(1)), 
                        np.maximum(df_1d['low'].shift(1) - df_1d['low'], 0), 0)
    tr_1d_for_adx = tr_1d  # reuse TR calculation
    plus_di_14 = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / pd.Series(tr_1d_for_adx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_di_14 = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / pd.Series(tr_1d_for_adx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dx_14 = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14 + 1e-10)
    adx_14 = pd.Series(dx_14).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(adx_14_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filters: only trade when volatility is elevated AND trend is strong enough
        vol_filter = atr_14_1d_aligned[i] > 0.003 * close[i]  # 0.3% of price
        trend_filter = adx_14_aligned[i] > 25  # Strong trend
        
        # Long conditions:
        # 1. Price above weekly EMA34 (bullish long-term bias)
        # 2. Volatility filter
        # 3. Trend strength filter
        if (close[i] > ema_34_1w_aligned[i] and
            vol_filter and
            trend_filter):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price below weekly EMA34 (bearish long-term bias)
        # 2. Volatility filter
        # 3. Trend strength filter
        elif (close[i] < ema_34_1w_aligned[i] and
              vol_filter and
              trend_filter):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

# Strategy using 12h timeframe with 1w trend filter + 1d volatility and ADX regime filters
# Works in bull markets (trend following) and bear markets (volatility + trend strength filters prevent whipsaws)
# Discrete position sizing (0.25) to minimize fee churn
name = "12h_EMA34_1w_Vol_ADX_Filter_v1"
timeframe = "12h"
leverage = 1.0