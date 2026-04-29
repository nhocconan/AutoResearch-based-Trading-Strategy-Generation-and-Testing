#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with weekly trend filter and volume confirmation
# Long when Williams %R < -80 (oversold) in bullish weekly trend (price > weekly EMA34) with volume spike
# Short when Williams %R > -20 (overbought) in bearish weekly trend (price < weekly EMA34) with volume spike
# Weekly trend filter avoids counter-trend trades in strong moves
# Volume confirmation ensures momentum participation
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag

name = "6h_WilliamsR_1wEMA34_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for weekly calculations
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    # Calculate weekly EMA(34) for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly EMA to 6h timeframe (completed weekly bar only)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Williams %R(14) on 6h
    williams_window = 14
    highest_high = pd.Series(high).rolling(window=williams_window, min_periods=williams_window).max().values
    lowest_low = pd.Series(low).rolling(window=williams_window, min_periods=williams_window).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: volume > 1.8x 24-period average (4 days on 6h)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (1.8 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(40, 24)  # warmup for weekly EMA and Williams %R
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema_34_1w_aligned[i]) or np.isnan(williams_r[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_williams = williams_r[i]
        curr_ema_trend = ema_34_1w_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Trend filter: bullish if price > weekly EMA34, bearish if price < weekly EMA34
        is_bullish_trend = curr_close > curr_ema_trend
        is_bearish_trend = curr_close < curr_ema_trend
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation
            if curr_volume_confirm:
                # Bullish entry: oversold in bullish trend
                if is_bullish_trend and curr_williams < -80:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: overbought in bearish trend
                elif is_bearish_trend and curr_williams > -20:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: Williams %R returns to oversold threshold OR breaks above -50 with volume
            if curr_williams >= -50 or (curr_williams > -80 and curr_volume_confirm):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: Williams %R returns to overbought threshold OR breaks below -50 with volume
            if curr_williams <= -50 or (curr_williams < -20 and curr_volume_confirm):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals