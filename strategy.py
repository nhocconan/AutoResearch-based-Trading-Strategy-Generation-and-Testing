#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Williams %R + 4h Trend Filter + Volume Confirmation
# Williams %R measures overbought/oversold levels. Readings below -80 indicate oversold, above -20 overbought.
# Uses 4h EMA(50) as trend filter to avoid counter-trend trades.
# Volume spike (1.5x 20-period average) confirms momentum.
# Entry: Long when Williams %R crosses above -80 in uptrend with volume spike.
#        Short when Williams %R crosses below -20 in downtrend with volume spike.
# Exit: Opposite Williams %R cross or trend reversal.
# Designed for 15-35 trades/year on 1h timeframe with focus on mean reversion in trending markets.
# Williams %R is effective in ranging markets while trend filter avoids major drawdowns.

name = "1h_4d_williams_r_volume_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(50) for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h EMA to 1h timeframe
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate volume moving average (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):  # Start after Williams %R period
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(williams_r[i-1]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Volume filter: current volume > 1.5 * 20-period average volume
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        # Determine 4h trend direction
        is_uptrend = close[i] > ema_50_4h_aligned[i]
        is_downtrend = close[i] < ema_50_4h_aligned[i]
        
        # Williams %R cross conditions
        wr_cross_above_80 = (williams_r[i-1] <= -80) and (williams_r[i] > -80)
        wr_cross_below_20 = (williams_r[i-1] >= -20) and (williams_r[i] < -20)
        
        # Entry conditions
        bullish_entry = wr_cross_above_80 and vol_filter and is_uptrend
        bearish_entry = wr_cross_below_20 and vol_filter and is_downtrend
        
        # Exit conditions: opposite Williams %R cross or trend reversal
        exit_long = wr_cross_below_20 or (position == 1 and not is_uptrend)
        exit_short = wr_cross_above_80 or (position == -1 and not is_downtrend)
        
        # Priority: entry > exit > hold
        if bullish_entry and position != 1:
            position = 1
            signals[i] = 0.20
        elif bearish_entry and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals