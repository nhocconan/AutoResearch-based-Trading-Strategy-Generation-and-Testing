#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d trend filter and volume confirmation
# - Long when price breaks above 12h Camarilla H3 level AND 1d close > 1d EMA50 (bullish trend)
# - Short when price breaks below 12h Camarilla L3 level AND 1d close < 1d EMA50 (bearish trend)
# - Volume confirmation: 12h volume > 1.3x 20-period volume SMA
# - Exit: price returns to 12h Camarilla PIVOT (mean reversion to center) or volume drops below average
# - Position sizing: 0.25 discrete level to minimize fee drag
# - Target: 12-37 trades/year on 12h timeframe to stay within fee drag limits
# - Uses proven Camarilla structure from top performers with proper 12h/1d alignment

name = "12h_1d_camarilla_breakout_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 12h Camarilla levels from previous 12h bar
    # Camarilla: H4 = close + 1.1*(high-low)*1.1/2, H3 = close + 1.1*(high-low)*1.1/4
    # L3 = close - 1.1*(high-low)*1.1/4, L4 = close - 1.1*(high-low)*1.1/2
    # Pivot = (high + low + close)/3
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    camarilla_pivot = (prev_high + prev_low + prev_close) / 3
    camarilla_range = prev_high - prev_low
    camarilla_h3 = camarilla_pivot + 1.1 * camarilla_range * 1.1 / 4
    camarilla_l3 = camarilla_pivot - 1.1 * camarilla_range * 1.1 / 4
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d close for trend comparison
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Calculate 12h volume SMA for regime filter
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(50, n):  # Start after warmup for indicators
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(camarilla_pivot[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(close_1d_aligned[i]) or np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 12h volume > 1.3x 20-period volume SMA
        vol_confirm = volume[i] > 1.3 * volume_sma_20[i]
        
        # Trend filter: 1d close vs 1d EMA50
        trend_bullish = close_1d_aligned[i] > ema_50_1d_aligned[i]
        trend_bearish = close_1d_aligned[i] < ema_50_1d_aligned[i]
        
        # Camarilla breakout signals
        breakout_up = close[i] > camarilla_h3[i]  # Break above H3 level
        breakout_down = close[i] < camarilla_l3[i]  # Break below L3 level
        
        # Exit conditions: return to pivot or loss of volume confirmation
        exit_long = close[i] < camarilla_pivot[i] or not vol_confirm
        exit_short = close[i] > camarilla_pivot[i] or not vol_confirm
        
        if position == 0:  # Flat - look for entry
            if breakout_up and trend_bullish and vol_confirm:
                position = 1
                signals[i] = 0.25
            elif breakout_down and trend_bearish and vol_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals