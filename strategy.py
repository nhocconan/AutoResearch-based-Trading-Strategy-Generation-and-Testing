#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d volume spike and 1d trend filter
# - Entry: Long when Williams %R(14) < -80 (oversold) + 1d volume > 1.5x 20-period average + 1d close > 1d EMA(50) (bullish trend)
#          Short when Williams %R(14) > -20 (overbought) + 1d volume > 1.5x 20-period average + 1d close < 1d EMA(50) (bearish trend)
# - Exit: Close-based mean reversion - exit long when Williams %R > -50, exit short when Williams %R < -50
# - Position sizing: 0.25 (discrete levels to minimize fee churn)
# - Uses Williams %R for mean reversion signals, volume for confirmation, EMA for trend filter
# - Designed for 6h timeframe to balance trade frequency and avoid fee drag, targeting 50-150 total trades over 4 years
# - Works in both bull and bear markets: mean reversion in ranges, trend filter avoids counter-trend in strong moves

name = "6h_1d_williamsr_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 6h OHLC
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    # Pre-compute 1d OHLC and volume
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Williams %R(14) on 6h
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high_14 - close_6h) / (highest_high_14 - lowest_low_14) * -100.0
    
    # Calculate 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d volume moving average (20-period)
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF data to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close_6h}), williams_r)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 6h close
        close_price = close_6h[i]
        
        # Get current 1d volume for confirmation (need to align raw volume)
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        volume_confirmation = volume_1d_aligned[i] > 1.5 * volume_ma_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Williams %R oversold + volume confirmation + bullish trend (close > EMA50)
            if (williams_r_aligned[i] < -80.0 and 
                volume_confirmation and 
                close_price > ema_50_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Williams %R overbought + volume confirmation + bearish trend (close < EMA50)
            elif (williams_r_aligned[i] > -20.0 and 
                  volume_confirmation and 
                  close_price < ema_50_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for mean reversion exit
            # Exit long when Williams %R > -50 (recovering from oversold)
            # Exit short when Williams %R < -50 (declining from overbought)
            if position == 1:
                if williams_r_aligned[i] > -50.0:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if williams_r_aligned[i] < -50.0:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals