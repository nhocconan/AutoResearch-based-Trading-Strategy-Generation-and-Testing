#!/usr/bin/env python3
name = "6h_VolatilityBreakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for trend and volatility
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily ATR(14) for volatility breakout
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.max([high_1d[0] - low_1d[0], tr2[0], tr3[0]]) if len(tr2) > 0 else high_1d[0] - low_1d[0]], tr])
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Daily EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily indicators to 6h timeframe
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 6h Bollinger Bands for volatility expansion
    bb_window = 20
    bb_std = 2.0
    sma_20 = pd.Series(close).rolling(window=bb_window, min_periods=bb_window).mean().values
    bb_std_dev = pd.Series(close).rolling(window=bb_window, min_periods=bb_window).std().values
    upper_bb = sma_20 + bb_std * bb_std_dev
    lower_bb = sma_20 - bb_std * bb_std_dev
    
    # Volume spike detection: 24-period average (4 days of 6h bars)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(bb_window, 50, 24)  # Wait for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(atr_14_1d_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(sma_20[i]) or np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Volatility breakout conditions
            bb_width = upper_bb[i] - lower_bb[i]
            vol_expansion = bb_width > np.nanmean(bb_width[max(0, i-48):i]) * 1.5 if i >= 48 else False
            vol_condition = volume[i] > vol_ma_24[i] * 2.0
            
            # Long: price breaks above upper BB with volatility expansion and daily uptrend
            if close[i] > upper_bb[i] and vol_expansion and vol_condition and ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower BB with volatility expansion and daily downtrend
            elif close[i] < lower_bb[i] and vol_expansion and vol_condition and ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns to middle BB or volatility contracts
            if close[i] < sma_20[i] or bb_width < np.nanmean(bb_width[max(0, i-48):i]) * 0.8 if i >= 48 else False:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to middle BB or volatility contracts
            if close[i] > sma_20[i] or bb_width < np.nanmean(bb_width[max(0, i-48):i]) * 0.8 if i >= 48 else False:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h volatility breakout with daily trend and volume confirmation
# - Uses daily ATR(14) and EMA(50) for trend/volatility context
# - 6h Bollinger Bands breakout with volatility expansion (BB width > 1.5x 48-bar average)
# - Requires volume spike (2x 24-bar average) for institutional confirmation
# - Long when price breaks above upper BB in daily uptrend with volume
# - Short when price breaks below lower BB in daily downtrend with volume
# - Exits when price returns to middle BB or volatility contracts
# - Works in both bull (breakouts in uptrend) and bear (breakdowns in downtrend)
# - Volatility filter avoids ranging markets, targets explosive moves
# - Position size 0.25 targets ~30-80 trades/year to avoid fee drag
# - Novel combination: volatility breakout (6h) + daily trend + volume confirmation
# - Avoids saturated BBands strategies by adding volatility expansion filter
# - Uses actual daily data via mtf_data for proper alignment (no look-ahead)
# - Designed for 6h timeframe to balance trade frequency and signal quality
# - Target: 50-150 total trades over 4 years (12-37/year) for BTC/ETH/USDT pairs
# - Daily trend filter reduces whipsaws vs pure Bollinger breakout
# - Volume confirmation reduces false breakouts in low liquidity periods