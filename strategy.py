#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d trend filter and volume spike
# - Williams %R(14) on 6h: oversold < -80, overbought > -20
# - 1d EMA50 trend filter: price > EMA50 = bullish bias, price < EMA50 = bearish bias
# - 6h volume spike: current volume > 1.5x 20-period average for confirmation
# - Long: Williams %R crosses above -80 (from below) AND 1d bullish trend AND volume spike
# - Short: Williams %R crosses below -20 (from above) AND 1d bearish trend AND volume spike
# - Exit: Williams %R crosses opposite threshold (-20 for longs, -80 for shorts) OR volume drops below average
# - Target: 12-30 trades/year on 6h (50-120 total over 4 years) to avoid fee drag

name = "6h_1d_williamsr_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Pre-compute 6h Williams %R(14)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.where((highest_high - lowest_low) != 0,
                          ((highest_high - close) / (highest_high - lowest_low)) * -100,
                          -50.0)  # neutral when range=0
    
    # Pre-compute 6h volume average (20-period)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(trend_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume spike condition: current volume > 1.5x 20-period average
        volume_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        # Williams %R conditions
        wr_now = williams_r[i]
        wr_prev = williams_r[i-1] if i > 0 else wr_now
        
        # Long: WR crosses above -80 from below
        long_entry = (wr_prev <= -80 and wr_now > -80)
        # Short: WR crosses below -20 from above
        short_entry = (wr_prev >= -20 and wr_now < -20)
        # Exit long: WR crosses above -20 (overbought)
        exit_long = (wr_prev < -20 and wr_now >= -20)
        # Exit short: WR crosses below -80 (oversold)
        exit_short = (wr_prev > -80 and wr_now <= -80)
        
        # 1d trend filter: price > EMA50 = bullish, price < EMA50 = bearish
        close_1d_current = df_1d['close'].values
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d_current)
        bullish_trend = not np.isnan(close_1d_aligned[i]) and not np.isnan(trend_aligned[i]) and \
                        close_1d_aligned[i] > trend_aligned[i]
        bearish_trend = not np.isnan(close_1d_aligned[i]) and not np.isnan(trend_aligned[i]) and \
                        close_1d_aligned[i] < trend_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: WR crosses above -80 AND bullish trend AND volume spike
            if long_entry and bullish_trend and volume_spike:
                position = 1
                signals[i] = 0.25
            # Short conditions: WR crosses below -20 AND bearish trend AND volume spike
            elif short_entry and bearish_trend and volume_spike:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: WR crosses opposite threshold OR volume drops below average
            vol_exhaustion = volume[i] < vol_ma_20[i]
            exit_condition = (position == 1 and exit_long) or (position == -1 and exit_short) or vol_exhaustion
            
            if exit_condition:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals