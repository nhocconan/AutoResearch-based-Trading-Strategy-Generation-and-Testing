#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with 1d EMA34 trend filter and volume confirmation.
# Williams %R measures overbought/oversold levels (-80 to -20). Long when %R crosses above -80 from below
# in uptrend (close > 1d EMA34) with volume > 1.3x 20-period MA. Short when %R crosses below -20 from above
# in downtrend (close < 1d EMA34) with volume spike. Uses discrete sizing 0.25 to minimize fee churn.
# Williams %R is effective in ranging markets which dominate 2025+ BTC/ETH price action.
# Target: 75-200 total trades over 4 years (19-50/year) with Sharpe > 0 on BTC/ETH/SOL.

name = "4h_WilliamsR_1dEMA34_Volume"
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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 4h Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Volume regime: current 4h volume > 1.3x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.3 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_34_1d_aligned[i]
        wr = williams_r[i]
        vol_spike = volume_spike[i]
        
        # Williams %R signals: cross above -80 (long), cross below -20 (short)
        wr_long_signal = wr > -80 and (i == 100 or williams_r[i-1] <= -80)
        wr_short_signal = wr < -20 and (i == 100 or williams_r[i-1] >= -20)
        
        # Determine trend regime
        is_uptrend = close_val > ema_trend
        is_downtrend = close_val < ema_trend
        
        # Entry logic
        if position == 0:
            if is_uptrend and wr_long_signal and vol_spike:
                signals[i] = 0.25
                position = 1
            elif is_downtrend and wr_short_signal and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses below -50 (momentum loss) OR trend reversal
            if wr < -50 or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses above -50 (momentum loss) OR trend reversal
            if wr > -50 or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals