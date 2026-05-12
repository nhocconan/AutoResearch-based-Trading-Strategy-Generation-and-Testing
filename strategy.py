#!/usr/bin/env python3
"""
6h_1d_1w_Iron_Condor_Credit_Spread
Hypothesis: Sell premium by selling OTM options equivalent via iron condor structure:
- Sell when price is between 1-week 1SD volatility bands (mean reversion in high IV)
- Buy protection at 2-week 2SD bands
- Uses 1d RSI(2) for mean reversion timing and 1w HV ratio for volatility regime
- Designed for range-bound markets (2025 BTC/ETH bearish/ranging) with defined risk
- Targets 15-25 trades/year by requiring low volatility environment + mean reversion signal
"""

name = "6h_1d_1w_Iron_Condor_Credit_Spread"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1d RSI(2) for short-term mean reversion (oversold/overbought)
    close_1d = pd.Series(close)
    delta = close_1d.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    avg_loss = loss.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_2 = 100 - (100 / (1 + rs))
    rsi_2 = rsi_2.fillna(50).values
    
    # 1d data for volatility calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # 1d log returns for volatility
    log_ret = np.log(df_1d['close'] / df_1d['close'].shift(1))
    # 10-day historical volatility (annualized)
    vol_10d = log_ret.rolling(window=10, min_periods=10).std() * np.sqrt(365)
    # 30-day historical volatility for regime
    vol_30d = log_ret.rolling(window=30, min_periods=30).std() * np.sqrt(365)
    # Volatility ratio: short-term/long-term vol (mean reversion signal)
    vol_ratio = vol_10d / vol_30d
    
    # 1w data for volatility bands (expected move)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # 1w close for center of bands
    wk_close = df_1w['close'].values
    # 1w ATM vol approximation from close-to-close
    wk_log_ret = np.log(wk_close / np.roll(wk_close, 1))
    wk_vol = pd.Series(wk_log_ret).rolling(window=4, min_periods=4).std() * np.sqrt(52)  # annualized
    # Convert to 6h vol: annual / sqrt(365*4) since 6 bars per day
    vol_6h = wk_vol / np.sqrt(365 * 4)
    
    # Calculate volatility bands (1SD and 2SD from weekly close)
    band_1sd = vol_6h * wk_close  # 1SD in price terms
    band_2sd = 2 * vol_6h * wk_close  # 2SD in price terms
    
    # Align all to 6h timeframe
    rsi_2_aligned = align_htf_to_ltf(prices, df_1d, rsi_2)
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio.values)
    band_1sd_aligned = align_htf_to_ltf(prices, df_1w, band_1sd.values)
    band_2sd_aligned = align_htf_to_ltf(prices, df_1w, band_2sd.values)
    wk_close_aligned = align_htf_to_ltf(prices, df_1w, wk_close)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long (bullish bias), -1: short (bearish bias)
    
    for i in range(30, n):
        if (np.isnan(rsi_2_aligned[i]) or
            np.isnan(vol_ratio_aligned[i]) or
            np.isnan(band_1sd_aligned[i]) or
            np.isnan(band_2sd_aligned[i]) or
            np.isnan(wk_close_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Calculate band levels
        upper_1sd = wk_close_aligned[i] + band_1sd_aligned[i]
        lower_1sd = wk_close_aligned[i] - band_1sd_aligned[i]
        upper_2sd = wk_close_aligned[i] + band_2sd_aligned[i]
        lower_2sd = wk_close_aligned[i] - band_2sd_aligned[i]
        
        # Mean reversion conditions:
        # - Low volatility regime (vol ratio < 0.8 = vol mean reverting down)
        # - RSI extreme for entry
        vol_regime = vol_ratio_aligned[i] < 0.8
        
        if position == 0:
            # LONG: Oversold + low vol regime + above 1SD lower band
            if (rsi_2_aligned[i] < 15 and 
                vol_regime and
                close[i] > lower_1sd):
                signals[i] = 0.25
                position = 1
            # SHORT: Overbought + low vol regime + below 1SD upper band
            elif (rsi_2_aligned[i] > 85 and 
                  vol_regime and
                  close[i] < upper_1sd):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI mean reversion or volatility expansion
            if (rsi_2_aligned[i] > 60 or 
                vol_ratio_aligned[i] > 1.2 or  # volatility expanding
                close[i] > upper_1sd):  # hit profit target
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI mean reversion or volatility expansion
            if (rsi_2_aligned[i] < 40 or 
                vol_ratio_aligned[i] > 1.2 or
                close[i] < lower_1sd):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals