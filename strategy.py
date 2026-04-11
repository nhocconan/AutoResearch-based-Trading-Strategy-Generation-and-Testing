#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_cci_volatility_v1"
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
    if len(df_1d) < 20:
        return signals
    
    # Calculate daily CCI(20) for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price
    tp_1d = (high_1d + low_1d + close_1d) / 3.0
    # SMA of typical price
    sma_tp_1d = pd.Series(tp_1d).rolling(window=20, min_periods=20).mean().values
    # Mean deviation
    md_1d = pd.Series(tp_1d).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    # CCI
    cci_1d = (tp_1d - sma_tp_1d) / (0.015 * md_1d)
    
    # Align CCI to 12h timeframe
    cci_1d_aligned = align_htf_to_ltf(prices, df_1d, cci_1d)
    
    # 12h Bollinger Bands for volatility breakout
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2.0 * std_20
    lower_bb = sma_20 - 2.0 * std_20
    
    # Volume confirmation: 12h volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(cci_1d_aligned[i]) or np.isnan(sma_20[i]) or np.isnan(std_20[i]) or 
            np.isnan(upper_bb[i]) or np.isnan(lower_bb[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        
        # Volume confirmation
        vol_confirm = volume_current > 1.3 * vol_ma_20[i]
        
        # Trend filter: CCI > 0 for long, CCI < 0 for short
        cci = cci_1d_aligned[i]
        trend_long = cci > 0
        trend_short = cci < 0
        
        # Volatility breakout entries
        breakout_up = price_close > upper_bb[i]
        breakout_down = price_close < lower_bb[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Upside breakout + bullish trend + volume confirmation
        if breakout_up and trend_long and vol_confirm:
            enter_long = True
        
        # Short: Downside breakout + bearish trend + volume confirmation
        if breakout_down and trend_short and vol_confirm:
            enter_short = True
        
        # Exit conditions: price crosses back to middle Bollinger Band
        exit_long = price_close < sma_20[i]
        exit_short = price_close > sma_20[i]
        
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

# Hypothesis: 12h Bollinger Band volatility breakout filtered by daily CCI(20) trend.
# In bull markets (CCI>0), we buy upside breakouts; in bear markets (CCI<0), we sell downside breakouts.
# Volume confirmation ensures institutional participation. The daily CCI filter provides multi-timeframe
# alignment, preventing counter-trend trades. Position size 0.25 manages drawdown. Target: 50-150 trades.