#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume spike and 1w trend filter
# - Long when price breaks above Camarilla H3 level AND 1d volume > 2x 20-period volume SMA AND 1w close > 1w EMA50
# - Short when price breaks below Camarilla L3 level AND 1d volume > 2x 20-period volume SMA AND 1w close < 1w EMA50
# - Exit: price returns to Camarilla PIVOT level or volume drops below average
# - Position sizing: 0.25 discrete level to minimize fee drag
# - Target: 12-30 trades/year on 12h timeframe to stay within fee drag limits
# - Why should work: Camarilla levels act as intraday support/resistance; volume confirms institutional interest; 1w trend filter avoids counter-trend trades

name = "12h_1d_1w_camarilla_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 12h Camarilla pivot levels from previous bar
    # Camarilla: PIVOT = (H+L+C)/3, Range = H-L
    # H4 = PIVOT + 1.1*(H-L)/2, H3 = PIVOT + 1.1*(H-L)/4, etc.
    # We use H3/L3 for breakouts
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    rang = prev_high - prev_low
    camarilla_h3 = pivot + 1.1 * rang / 4.0
    camarilla_l3 = pivot - 1.1 * rang / 4.0
    camarilla_pivot = pivot  # For exit
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1w close for trend comparison
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    
    # Calculate 1d volume SMA for regime filter
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(50, n):  # Start after warmup for indicators
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(camarilla_pivot[i]) or np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(close_1w_aligned[i]) or np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 12h volume > 2x 20-period volume SMA (strict for low trade count)
        vol_confirm = volume[i] > 2.0 * volume_sma_20[i]
        
        # Trend filter: 1w close vs 1w EMA50
        trend_bullish = close_1w_aligned[i] > ema_50_1w_aligned[i]
        trend_bearish = close_1w_aligned[i] < ema_50_1w_aligned[i]
        
        # Camarilla breakout signals
        breakout_up = close[i] > camarilla_h3[i]  # Break above H3
        breakout_down = close[i] < camarilla_l3[i]  # Break below L3
        
        # Exit conditions: return to pivot or loss of volume confirmation
        exit_long = (close[i] < camarilla_pivot[i]) or not vol_confirm
        exit_short = (close[i] > camarilla_pivot[i]) or not vol_confirm
        
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