#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return signals
    
    # Calculate daily ATR for volatility filter (14-period)
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
    
    # Calculate daily ATR MA (50-period) for trend filter
    atr_ma_50_1d = pd.Series(atr_14_1d_aligned).rolling(window=50, min_periods=50).mean().values
    
    # Calculate Camarilla levels on daily data (using previous day's range)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    camarilla_H4_1d = prev_close_1d + 1.1 * (prev_high_1d - prev_low_1d) / 2
    camarilla_L4_1d = prev_close_1d - 1.1 * (prev_high_1d - prev_low_1d) / 2
    camarilla_C_1d = prev_close_1d  # Camarilla C level is previous close
    
    # Align Camarilla levels to 12h timeframe
    camarilla_H4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H4_1d)
    camarilla_L4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L4_1d)
    camarilla_C_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_C_1d)
    
    # Volume confirmation: 20-period average on 12h
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_H4_1d_aligned[i]) or np.isnan(camarilla_L4_1d_aligned[i]) or
            np.isnan(camarilla_C_1d_aligned[i]) or np.isnan(volume_sma_20[i]) or
            np.isnan(atr_ratio_1d[i]) or np.isnan(atr_ma_50_1d[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average (stricter filter)
        vol_confirm = volume_current > 2.0 * volume_sma_20[i]
        
        # Trend filter: trade only when ATR is above its 50-period MA (trending market)
        trending = atr_14_1d_aligned[i] > atr_ma_50_1d[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Price breaks above Camarilla H4 level + volume confirmation + trending
        price_above_H4 = price_close > camarilla_H4_1d_aligned[i]
        if price_above_H4 and vol_confirm and trending:
            enter_long = True
        
        # Short: Price breaks below Camarilla L4 level + volume confirmation + trending
        price_below_L4 = price_close < camarilla_L4_1d_aligned[i]
        if price_below_L4 and vol_confirm and trending:
            enter_short = True
        
        # Exit conditions: price crosses back through the Camarilla C level (previous day's close)
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price crosses below Camarilla C level
            exit_long = price_close < camarilla_C_1d_aligned[i]
        elif position == -1:
            # Exit short if price crosses above Camarilla C level
            exit_short = price_close > camarilla_C_1d_aligned[i]
        
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

# Hypothesis: Camarilla breakout on 12h timeframe with daily trend and volatility filters.
# Uses daily Camarilla levels (H4/L4) for entry and C level (previous close) for exit.
# Volume confirmation (>2.0x 20-period average) ensures strong institutional participation.
# ATR trend filter (ATR > 50-period MA) ensures we only trade in trending markets.
# Target: 20-40 trades/year to minimize fee drift while capturing major moves.