#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d trend filter and volume confirmation
# - Williams %R(14) measures overbought/oversold levels (-80 to -20 = oversold, -20 to 0 = overbought)
# - Long when Williams %R < -80 (oversold) AND 1d close > EMA50 AND volume > 1.5x avg
# - Short when Williams %R > -20 (overbought) AND 1d close < EMA50 AND volume > 1.5x avg
# - Exit when Williams %R returns to -50 (mean reversion midpoint) or opposite extreme reached
# - Uses discrete position sizing (0.25) to control drawdown
# - Targets ~15-25 trades/year (60-100 total over 4 years) to avoid fee drag
# - Williams %R is effective in ranging markets and captures reversals from extremes
# - 1d EMA50 filter ensures alignment with higher timeframe trend
# - Volume confirmation prevents false signals in low-volume environments

name = "6h_1d_williamsr_meanreversion_volume_trend_v2"
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
    
    # Pre-compute Williams %R(14) for 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate highest high and lowest low over 14 periods
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        ((highest_high - close) / (highest_high - lowest_low)) * -100,
        -50  # Neutral when range is zero
    )
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(volume_20_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long signal: Williams %R oversold (< -80) AND 1d uptrend AND volume spike
            if (williams_r[i] < -80 and 
                close[i] > ema50_1d_aligned[i] and 
                vol_spike.iloc[i]):
                position = 1
                signals[i] = 0.25
            # Short signal: Williams %R overbought (> -20) AND 1d downtrend AND volume spike
            elif (williams_r[i] > -20 and 
                  close[i] < ema50_1d_aligned[i] and 
                  vol_spike.iloc[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Williams %R returns to midpoint (-50) = mean reversion complete
            # 2. Opposite extreme reached (long exits at > -20, short exits at < -80)
            if position == 1:
                if williams_r[i] >= -50 or williams_r[i] > -20:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:
                if williams_r[i] <= -50 or williams_r[i] < -80:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals