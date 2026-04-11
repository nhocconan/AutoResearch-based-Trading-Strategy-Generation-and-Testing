#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_1w_vwap_reversion_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily and weekly data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 20 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate daily VWAP
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap_1d = (typical_price_1d * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_1d = vwap_1d.values
    
    # Calculate weekly VWAP
    typical_price_1w = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    vwap_1w = (typical_price_1w * df_1w['volume']).cumsum() / df_1w['volume'].cumsum()
    vwap_1w = vwap_1w.values
    
    # Shift by 1 to use only completed daily/weekly bars
    vwap_1d = np.roll(vwap_1d, 1)
    vwap_1w = np.roll(vwap_1w, 1)
    vwap_1d[0] = np.nan
    vwap_1w[0] = np.nan
    
    # Align daily and weekly VWAP to 4h timeframe
    vwap_1d_4h = align_htf_to_ltf(prices, df_1d, vwap_1d)
    vwap_1w_4h = align_htf_to_ltf(prices, df_1w, vwap_1w)
    
    # 4h ATR for volatility filter (14 period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 4h volume filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Daily volume ratio filter (volume > 2x daily average)
    vol_1d = df_1d['volume'].values
    vol_ma_10_1d = pd.Series(vol_1d).rolling(window=10, min_periods=10).mean().values
    vol_ma_10_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_10_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(vwap_1d_4h[i]) or np.isnan(vwap_1w_4h[i]) or
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(vol_ma_10_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        vol_daily_ma = vol_ma_10_1d_aligned[i]
        
        # Volume confirmation: both 4h and daily volume must be elevated
        volume_confirmed = (volume_current > 1.5 * vol_ma) and (volume_current > 2.0 * vol_daily_ma)
        
        # Long conditions: price below VWAPs with volume and price above weekly VWAP (mean reversion)
        long_signal = volume_confirmed and (price_close < vwap_1d_4h[i]) and (price_close > vwap_1w_4h[i])
        
        # Short conditions: price above VWAPs with volume and price below weekly VWAP (mean reversion)
        short_signal = volume_confirmed and (price_close > vwap_1d_4h[i]) and (price_close < vwap_1w_4h[i])
        
        # Exit when price returns to daily VWAP (mean reversion)
        exit_long = position == 1 and price_close >= vwap_1d_4h[i]
        exit_short = position == -1 and price_close <= vwap_1d_4h[i]
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: 4h VWAP reversion with dual timeframe VWAP and volume confirmation.
# Uses daily and weekly VWAP levels to identify mean reversion opportunities.
# Enters long when 4h price is below daily VWAP (oversold) but above weekly VWAP (long-term uptrend intact)
# with volume confirmation (>1.5x 4h 20-period average and >2x daily 10-period average).
# Enters short when 4h price is above daily VWAP (overbought) but below weekly VWAP (long-term downtrend intact)
# with same volume conditions. Exits when price returns to daily VWAP.
# Daily VWAP provides short-term mean reversion level, weekly VWAP provides trend filter.
# Volume confirmation filters out low-volume false signals.
# Position size: 0.25 to balance risk and return, limiting drawdown in volatile markets.
# Designed to work in both bull and bear markets by combining mean reversion with trend filter.
# Target: 30-80 trades over 4 years (7-20/year) to minimize fee drag.