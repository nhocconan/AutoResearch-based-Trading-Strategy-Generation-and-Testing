#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Camarilla pivot levels with volume confirmation and ATR-based trend filter
# Camarilla pivots provide structured support/resistance levels based on previous day's range
# Long when price breaks above H3 with volume confirmation in bullish trend (close > EMA50)
# Short when price breaks below L3 with volume confirmation in bearish trend (close < EMA50)
# In sideways markets (price near EMA50), no new entries to avoid whipsaw
# Uses discrete position sizing 0.25 to target ~15-35 trades/year and minimize fee drag
# Works in bull/bear markets: breakout follows trends, avoids counter-trend entries in ranging conditions

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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values if 'volume' in df_1d.columns else np.zeros_like(close_1d)
    
    # Calculate 1d Camarilla pivot levels
    range_1d = high_1d - low_1d
    camarilla_h3 = close_1d + 1.1 * range_1d
    camarilla_l3 = close_1d - 1.1 * range_1d
    camarilla_h4 = close_1d + 1.5 * range_1d
    camarilla_l4 = close_1d - 1.5 * range_1d
    
    # Calculate 1d EMA50 for trend filter
    close_s_1d = pd.Series(close_1d)
    ema50_1d = close_s_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d average volume (20-period)
    vol_s_1d = pd.Series(volume_1d)
    avg_vol_1d = vol_s_1d.rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(avg_vol_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_confirmed = not np.isnan(vol_ma_20[i]) and volume[i] > 1.5 * vol_ma_20[i]
        
        # Trend filter: bullish if close > EMA50, bearish if close < EMA50
        bullish_trend = close[i] > ema50_1d_aligned[i]
        bearish_trend = close[i] < ema50_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit if price falls below L3 (mean reversion) or trend turns bearish
            if close[i] < camarilla_l3_aligned[i] or not bullish_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if price rises above H3 (mean reversion) or trend turns bullish
            if close[i] > camarilla_h3_aligned[i] or not bearish_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Only enter on breakouts with volume confirmation and trend alignment
            if volume_confirmed:
                if bullish_trend and close[i] > camarilla_h3_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                elif bearish_trend and close[i] < camarilla_l3_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals