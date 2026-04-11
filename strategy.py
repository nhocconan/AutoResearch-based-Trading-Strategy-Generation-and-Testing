# 12h_1d_1w_camarilla_volume_v3 - 12h timeframe with 1d/1w confluence
# Hypothesis: Multi-timeframe confluence of weekly trend + daily Camarilla levels + volume confirmation on 12h timeframe
# Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag
# Works in bull/bear by requiring weekly trend alignment with daily breakouts
# Uses 1d Camarilla levels for entry/exit and 1w EMA200 for trend filter

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_1w_camarilla_volume_v3"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 500:  # Need sufficient history for weekly indicators
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return signals
    
    # Calculate 1w EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Calculate daily ATR for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr_14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate daily ATR MA (20-period) for volatility regime filter
    atr_ma_20_1d = pd.Series(atr_14_1d_aligned).rolling(window=20, min_periods=20).mean().values
    
    # Calculate daily ATR ratio (current / MA) for regime detection
    atr_ratio_1d = np.where(atr_ma_20_1d > 0, atr_14_1d_aligned / atr_ma_20_1d, 1.0)
    
    # Calculate Camarilla levels on daily data (using previous daily bar's range)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    camarilla_H4 = prev_close_1d + 1.1 * (prev_high_1d - prev_low_1d) / 2
    camarilla_L4 = prev_close_1d - 1.1 * (prev_high_1d - prev_low_1d) / 2
    camarilla_C = prev_close_1d  # C level is previous day's close
    
    # Align Camarilla levels to 12h timeframe
    camarilla_H4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H4)
    camarilla_L4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L4)
    camarilla_C_aligned = align_htf_to_ltf(prices, df_1d, camarilla_C)
    
    # Volume confirmation: 30-period average on 12h
    volume_sma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    for i in range(500, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_H4_aligned[i]) or np.isnan(camarilla_L4_aligned[i]) or
            np.isnan(camarilla_C_aligned[i]) or np.isnan(volume_sma_30[i]) or
            np.isnan(ema200_1w_aligned[i]) or np.isnan(atr_ratio_1d[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 2.0x 30-period average
        vol_confirm = volume_current > 2.0 * volume_sma_30[i]
        
        # Volatility regime filter: trade only when volatility is elevated (ATR ratio > 0.6)
        vol_regime = atr_ratio_1d[i] > 0.6
        
        # Weekly trend filter
        weekly_uptrend = price_close > ema200_1w_aligned[i]
        weekly_downtrend = price_close < ema200_1w_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Price breaks above Camarilla H4 + volume + volatility + weekly uptrend
        price_above_H4 = price_close > camarilla_H4_aligned[i]
        if price_above_H4 and vol_confirm and vol_regime and weekly_uptrend:
            enter_long = True
        
        # Short: Price breaks below Camarilla L4 + volume + volatility + weekly downtrend
        price_below_L4 = price_close < camarilla_L4_aligned[i]
        if price_below_L4 and vol_confirm and vol_regime and weekly_downtrend:
            enter_short = True
        
        # Exit conditions: price crosses back through the Camarilla C level
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price crosses below Camarilla C level
            exit_long = price_close < camarilla_C_aligned[i]
        elif position == -1:
            # Exit short if price crosses above Camarilla C level
            exit_short = price_close > camarilla_C_aligned[i]
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Multi-timeframe confluence of weekly trend + daily Camarilla levels + volume confirmation on 12h timeframe
# Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag
# Works in bull/bear by requiring weekly trend alignment with daily breakouts
# Uses 1d Camarilla levels for entry/exit and 1w EMA200 for trend filter
# Volume confirmation >2.0x average and ATR ratio >0.6 ensures institutional participation in volatile conditions
# Conservative sizing (0.25) to manage drawdown in volatile markets like 2022 and 2025+