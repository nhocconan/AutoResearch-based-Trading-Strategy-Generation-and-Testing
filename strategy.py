#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Bollinger Band squeeze breakout with weekly volume confirmation.
# Bollinger squeeze identifies low volatility periods that precede breakouts.
# Volume confirmation on breakout ensures institutional participation.
# Weekly trend filter aligns with higher timeframe direction.
# Target: 15-30 trades per year (60-120 total over 4 years) for 1d timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate Bollinger Bands (20, 2)
    close_1d = df_1d['close'].values
    sma_20 = np.full(len(close_1d), np.nan)
    std_20 = np.full(len(close_1d), np.nan)
    
    for i in range(19, len(close_1d)):
        sma_20[i] = np.mean(close_1d[i-19:i+1])
        std_20[i] = np.std(close_1d[i-19:i+1])
    
    upper = sma_20 + 2 * std_20
    lower = sma_20 - 2 * std_20
    bb_width = (upper - lower) / sma_20
    
    # Bollinger squeeze: BB width below 20-period percentile
    bb_width_pct = np.full(len(bb_width), np.nan)
    for i in range(39, len(bb_width)):
        bb_width_pct[i] = np.percentile(bb_width[i-39:i+1], 25)
    
    squeeze = bb_width < bb_width_pct
    
    # Align squeeze signal to lower timeframe
    squeeze_aligned = align_htf_to_ltf(prices, df_1d, squeeze.astype(float))
    
    # Calculate weekly EMA trend filter
    close_1w = df_1w['close'].values
    ema_10 = np.zeros(len(close_1w))
    ema_multiplier = 2 / (10 + 1)
    ema_10[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        ema_10[i] = (close_1w[i] - ema_10[i-1]) * ema_multiplier + ema_10[i-1]
    
    ema_10_aligned = align_htf_to_ltf(prices, df_1w, ema_10)
    
    # Calculate average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(19, n):
        avg_volume[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(40, n):
        # Skip if any required data is not ready
        if (np.isnan(squeeze_aligned[i]) or np.isnan(ema_10_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        squeeze_signal = squeeze_aligned[i]
        weekly_ema = ema_10_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: breakout above upper BB with volume + above weekly EMA
            if (price > upper[-1] if hasattr(upper, '__len__') else price > upper[i] if i < len(upper) else False) and \
               volume_confirm and \
               price > weekly_ema:
                # Need to get upper band value for current day
                day_idx = i // 24  # approximate days
                if day_idx < len(upper) and not np.isnan(upper[day_idx]):
                    if price > upper[day_idx] and volume_confirm and price > weekly_ema:
                        position = 1
                        signals[i] = position_size
            # Short: breakout below lower BB with volume + below weekly EMA
            elif price < lower[-1] if hasattr(lower, '__len__') else price < lower[i] if i < len(lower) else False:
                day_idx = i // 24
                if day_idx < len(lower) and not np.isnan(lower[day_idx]):
                    if price < lower[day_idx] and volume_confirm and price < weekly_ema:
                        position = -1
                        signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to middle BB or squeeze ends
            day_idx = i // 24
            if day_idx < len(sma_20) and not np.isnan(sma_20[day_idx]):
                if price < sma_20[day_idx] or squeeze_signal < 0.5:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = position_size
            else:
                signals[i] = 0.0
        elif position == -1:
            # Exit short: price returns to middle BB or squeeze ends
            day_idx = i // 24
            if day_idx < len(sma_20) and not np.isnan(sma_20[day_idx]):
                if price > sma_20[day_idx] or squeeze_signal < 0.5:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_20_Bollinger_Squeeze_Breakout_Volume_Trend_v1"
timeframe = "1d"
leverage = 1.0