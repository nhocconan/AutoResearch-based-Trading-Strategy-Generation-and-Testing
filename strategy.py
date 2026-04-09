#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Williams %R with volume confirmation and ATR-based trend filter
# Williams %R identifies overbought/oversold conditions on 12h timeframe
# Long when %R crosses above -80 (oversold) with volume confirmation and ATR indicating trend strength
# Short when %R crosses below -20 (overbought) with volume confirmation and ATR indicating trend strength
# Uses discrete position sizing 0.25 to target ~20-50 trades/year and minimize fee drag
# Works in bull/bear markets: mean reversion at extremes in ranging markets, momentum in trending regimes

name = "4h_12h_williamsr_volume_trend_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values if 'volume' in df_12h.columns else np.zeros_like(close_12h)
    
    # Calculate 12h Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        ((highest_high - close_12h) / (highest_high - lowest_low)) * -100,
        -50  # neutral when range is zero
    )
    
    # Calculate 12h ATR(14) for trend strength filter
    tr1 = np.abs(high_12h[1:] - low_12h[:-1])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_12h = wilders_smoothing(tr, 14)
    atr_ma_12h = pd.Series(atr_12h).rolling(window=10, min_periods=10).mean().values
    
    # Calculate 12h average volume (20-period)
    vol_s_12h = pd.Series(volume_12h)
    avg_vol_12h = vol_s_12h.rolling(window=20, min_periods=20).mean().values
    
    # Align 12h indicators to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    avg_vol_12h_aligned = align_htf_to_ltf(prices, df_12h, avg_vol_12h)
    atr_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r_aligned[i]) or np.isnan(avg_vol_12h_aligned[i]) or
            np.isnan(atr_ma_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.3x 20-period average
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_confirmed = volume[i] > 1.3 * vol_ma_20[i] if not np.isnan(vol_ma_20[i]) else False
        
        # Trend filter: ATR above its moving average indicates strengthening trend
        trend_strong = atr_12h_aligned[i] > atr_ma_12h_aligned[i] if not np.isnan(atr_12h_aligned[i]) and not np.isnan(atr_ma_12h_aligned[i]) else True
        
        if position == 1:  # Long position
            # Exit long if Williams %R rises above -20 (overbought) or volume dries up
            if williams_r_aligned[i] > -20 or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit short if Williams %R falls below -80 (oversold) or volume dries up
            if williams_r_aligned[i] < -80 or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long when Williams %R crosses above -80 from below (oversold bounce)
            # Enter short when Williams %R crosses below -20 from above (overbought rejection)
            if i > 100:  # ensure we have previous value
                prev_williams = williams_r_aligned[i-1]
                curr_williams = williams_r_aligned[i]
                
                if (prev_williams <= -80 and curr_williams > -80 and 
                    volume_confirmed and trend_strong):
                    position = 1
                    signals[i] = 0.25
                elif (prev_williams >= -20 and curr_williams < -20 and 
                      volume_confirmed and trend_strong):
                    position = -1
                    signals[i] = -0.25
    
    return signals