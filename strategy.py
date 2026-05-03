#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
# Long when price breaks above Camarilla R3 level in bull trend (close > 1d EMA34) with volume > 1.5x 20-period MA.
# Short when price breaks below Camarilla S3 level in bear trend (close < 1d EMA34) with volume spike.
# Uses discrete position sizing (0.30) to balance return and fee drag.
# 1d EMA34 provides higher timeframe trend filter to avoid counter-trend trades.
# Volume confirmation ensures moves have institutional participation.
# Target: 75-200 total trades over 4 years (19-50/year) with Sharpe > 0 on BTC/ETH/SOL.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Volume"
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
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous day
    # Camarilla levels: based on previous day's high, low, close
    # R3 = C + (H-L) * 1.1/2
    # S3 = C - (H-L) * 1.1/2
    # We need to use previous day's data to avoid look-ahead
    prev_high = np.roll(high, 1)  # previous bar's high
    prev_low = np.roll(low, 1)    # previous bar's low
    prev_close = np.roll(close, 1) # previous bar's close
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    camarilla_range = (prev_high - prev_low) * 1.1 / 2
    camarilla_r3 = prev_close + camarilla_range
    camarilla_s3 = prev_close - camarilla_range
    
    # Volume regime: current 4h volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_34_1d_aligned[i]
        r3_level = camarilla_r3[i]
        s3_level = camarilla_s3[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_bull_trend = close_val > ema_trend
        is_bear_trend = close_val < ema_trend
        
        # Entry logic
        if position == 0:
            # Long: break above R3 in bull trend with volume spike
            if is_bull_trend and close_val > r3_level and vol_spike:
                signals[i] = 0.30
                position = 1
            # Short: break below S3 in bear trend with volume spike
            elif is_bear_trend and close_val < s3_level and vol_spike:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long exit: price below S3 (reversal) or trend reversal
            if close_val < s3_level or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short exit: price above R3 (reversal) or trend reversal
            if close_val > r3_level or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals