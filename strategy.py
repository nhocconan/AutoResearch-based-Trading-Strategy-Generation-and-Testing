#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1w trend filter and volume confirmation.
# Williams %R measures overbought/oversold levels: (Highest High - Close) / (Highest High - Lowest Low) * -100
# Long when WR < -80 (oversold) and rising, price above 1w EMA50 (bullish trend).
# Short when WR > -20 (overbought) and falling, price below 1w EMA50 (bearish trend).
# Volume confirmation: current volume > 1.3x 20-period average.
# Uses mean reversion in extremes with trend filter to avoid counter-trend trades.
# Target: 75-150 total trades over 4 years (19-38/year).

name = "6h_williamsr_trend_filter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R (14-period)
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # Slope of Williams %R (1-period change)
    wr_slope = np.diff(williams_r, prepend=williams_r[0])
    
    # 1w trend filter: 50-period EMA on weekly chart
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50w_aligned = align_htf_to_ltf(prices, df_1w, ema_50w)
    
    # Volume filter: current volume > 1.3x 20-period average
    volume_s = pd.Series(volume)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(14, n):
        # Skip if 1w trend data not available
        if np.isnan(ema_50w_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.3
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: WR rises above -50 or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price - 2.5 * atr_approx
            
            if (williams_r[i] >= -50 or 
                close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: WR falls below -50 or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price + 2.5 * atr_approx
            
            if (williams_r[i] <= -50 or 
                close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and 1w trend filter
            if volume_filter:
                # Long: WR < -80 (oversold) and rising, price above 1w EMA50
                if (williams_r[i] < -80 and wr_slope[i] > 0 and 
                    close[i] > ema_50w_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: WR > -20 (overbought) and falling, price below 1w EMA50
                elif (williams_r[i] > -20 and wr_slope[i] < 0 and 
                      close[i] < ema_50w_aligned[i]):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals