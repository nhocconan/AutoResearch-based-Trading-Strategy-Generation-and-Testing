#!/usr/bin/env python3
# Hypothesis: 4h Camarilla H4/L4 breakout with 1d RSI regime filter and 4h volume confirmation.
# Long when price breaks above H4 with 1d RSI > 50 (bullish regime) and 4h volume > 2.0x 20-period average.
# Short when price breaks below L4 with 1d RSI < 50 (bearish regime) and 4h volume > 2.0x 20-period average.
# Exit on opposite Camarilla level (L4 for longs, H4 for shorts).
# Uses discrete position sizing (0.25) to minimize fee churn. RSI regime filter ensures trend alignment,
# reducing false breakouts in ranging markets. Target: 75-200 total trades over 4 years = 19-50/year for 4h.
# Works in bull/bear: 1d RSI confirms market regime, Camarilla H4/L4 provides tight structure, volume confirms momentum.

name = "4h_Camarilla_H4L4_Breakout_1dRSI_4hVolumeConfirm"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 4h Indicators (LTF) ---
    # 4h volume confirmation: > 2.0x 20-period average (tight filter to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm_4h = volume > (2.0 * vol_ma_20)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # 1d RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14 = 100 - (100 / (1 + rs))
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14)
    
    # --- 4h Camarilla Pivot Points (Prior Day OHLC) ---
    camarilla_h4 = np.full(n, np.nan)
    camarilla_l4 = np.full(n, np.nan)
    df_1d_pivot = get_htf_data(prices, '1d')
    if len(df_1d_pivot) == 0:
        return np.zeros(n)
    
    # Precompute prior day's OHLC for each 4h bar
    open_time = prices['open_time']
    prior_day_start = open_time - pd.Timedelta(days=1)
    prior_day_start = prior_day_start.dt.normalize()  # Start of prior day
    
    for i in range(n):
        pd_ts = prior_day_start.iloc[i]
        day_mask = (df_1d_pivot['open_time'] >= pd_ts) & (df_1d_pivot['open_time'] < pd_ts + pd.Timedelta(days=1))
        if day_mask.any():
            day_data = df_1d_pivot.loc[day_mask]
            high_val = day_data['high'].iloc[0]
            low_val = day_data['low'].iloc[0]
            close_val = day_data['close'].iloc[0]
            range_val = high_val - low_val
            camarilla_h4[i] = close_val + (range_val * 1.1 / 2)  # H4
            camarilla_l4[i] = close_val - (range_val * 1.1 / 2)  # L4
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(rsi_14_aligned[i]) or
            np.isnan(volume_confirm_4h[i]) or
            np.isnan(camarilla_h4[i]) or
            np.isnan(camarilla_l4[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above H4 + 1d RSI > 50 (bullish) + 4h volume confirmation
            if (close[i] > camarilla_h4[i] and 
                rsi_14_aligned[i] > 50 and 
                volume_confirm_4h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below L4 + 1d RSI < 50 (bearish) + 4h volume confirmation
            elif (close[i] < camarilla_l4[i] and 
                  rsi_14_aligned[i] < 50 and 
                  volume_confirm_4h[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below L4
            if close[i] < camarilla_l4[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above H4
            if close[i] > camarilla_h4[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals