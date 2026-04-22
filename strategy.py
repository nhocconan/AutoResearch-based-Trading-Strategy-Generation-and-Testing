#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Bollinger Band squeeze breakout with 1w trend filter and 1d volume confirmation
# Bollinger Band squeeze (BB width at 20-period low) indicates low volatility and imminent breakout.
# Direction determined by 1w EMA34 trend (bullish if close > EMA34, bearish if close < EMA34).
# Entry confirmed by 1d volume spike (> 1.5x 20-day average) to avoid false breakouts.
# Works in bull markets by capturing breakouts upward and in bear markets by avoiding false breakdowns
# via trend filter and requiring volume confirmation. Designed for 1d timeframe targeting 10-25 trades/year.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for BB calculations and volume confirmation (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Load 1w data for trend filter (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # 1w EMA(34) for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # 1d volume 20-period average for spike detection
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    # Bollinger Bands (20, 2) on 1d data
    sma_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20_1d + 2 * std_20_1d
    lower_bb = sma_20_1d - 2 * std_20_1d
    bb_width = upper_bb - lower_bb
    
    # Bollinger Band squeeze: BB width at 20-period low
    bb_width_low_20 = pd.Series(bb_width).rolling(window=20, min_periods=20).min().values
    bb_squeeze = bb_width <= bb_width_low_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_avg_20_1d_aligned[i]) or
            np.isnan(sma_20_1d[i]) or np.isnan(std_20_1d[i]) or
            np.isnan(bb_width_low_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: BB squeeze + 1w uptrend + 1d volume spike
            if (bb_squeeze[i] and 
                close_1d[i] > ema_34_1w_aligned[i] and 
                volume_1d[i] > 1.5 * vol_avg_20_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: BB squeeze + 1w downtrend + 1d volume spike
            elif (bb_squeeze[i] and 
                  close_1d[i] < ema_34_1w_aligned[i] and 
                  volume_1d[i] > 1.5 * vol_avg_20_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price returns to SMA20 or trend reversal
            if position == 1:
                # Exit on return to SMA20 or trend reversal
                if (close_1d[i] <= sma_20_1d[i] or 
                    close_1d[i] < ema_34_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit on return to SMA20 or trend reversal
                if (close_1d[i] >= sma_20_1d[i] or 
                    close_1d[i] > ema_34_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_BBSqueeze_1wEMA34_1dVolSpike"
timeframe = "1d"
leverage = 1.0