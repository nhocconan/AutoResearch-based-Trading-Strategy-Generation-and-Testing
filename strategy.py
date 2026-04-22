#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Bollinger Band squeeze breakout with 4h trend filter and 1d volume confirmation
# Bollinger Band squeeze (BB width at 20-period low) indicates low volatility and imminent breakout.
# Direction determined by 4h EMA50 trend (bullish if close > EMA50, bearish if close < EMA50).
# Entry confirmed by 1d volume spike (> 1.5x 20-day average) to avoid false breakouts.
# Works in bull markets by capturing breakouts upward and in bear markets by avoiding false breakdowns
# via trend filter and requiring volume confirmation. Designed for 1h timeframe targeting 15-35 trades/year.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data for trend filter (ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Load 1d data for volume confirmation (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # 4h EMA(50) for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d volume 20-period average for spike detection
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    # Bollinger Bands (20, 2) on 1h data
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    bb_width = upper_bb - lower_bb
    
    # Bollinger Band squeeze: BB width at 20-period low
    bb_width_low_20 = pd.Series(bb_width).rolling(window=20, min_periods=20).min().values
    bb_squeeze = bb_width <= bb_width_low_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_avg_20_1d_aligned[i]) or
            np.isnan(sma_20[i]) or np.isnan(std_20[i]) or
            np.isnan(bb_width_low_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: BB squeeze + 4h uptrend + 1d volume spike
            if (bb_squeeze[i] and 
                close[i] > ema_50_4h_aligned[i] and 
                volume[i] > 1.5 * vol_avg_20_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: BB squeeze + 4h downtrend + 1d volume spike
            elif (bb_squeeze[i] and 
                  close[i] < ema_50_4h_aligned[i] and 
                  volume[i] > 1.5 * vol_avg_20_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions: price returns to SMA20 or trend reversal
            if position == 1:
                # Exit on return to SMA20 or trend reversal
                if (close[i] <= sma_20[i] or 
                    close[i] < ema_50_4h_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                # Exit on return to SMA20 or trend reversal
                if (close[i] >= sma_20[i] or 
                    close[i] > ema_50_4h_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals

name = "1h_BBSqueeze_4hEMA50_1dVolSpike"
timeframe = "1h"
leverage = 1.0